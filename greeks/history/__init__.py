"""Historical greeks on every option_trade row (trade basis), written back into option_trade.

Every row's IV + greeks come from the ONE shared core (greeks.pricing.price_options): the trade price
inverted at trade_ts against spot_at_trade (the respot corrected it). Live greeks price the same way
off a quote mid — only the basis differs, recorded per row (greeks_basis='trade' here). This module
has no pricing logic of its own; it parses the OSI, checks spot, and hands the chunk to the core.

The rewrite follows RespotJob, per (day, underlying): read the day's rows (FINAL), load the day's
inputs once, price, re-insert the FULL rows with the greeks columns filled and a fresh ingested_at
(ReplacingMergeTree(ingested_at) keeps the new version) — every other column copied exactly — then
write an option_archive_ledger row with dataset='greeks_<methodology>'. A rerun skips pairs that
already have that ledger row, so the job is resumable and safe to re-run. Workers are independent
processes (one (day, underlying) each) — no asyncio/threading under greeks/.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from greeks.trades import GREEKS_COLUMNS, Inputs, TradeRow, price_trade_rows

log = logging.getLogger(__name__)

Pair = Tuple[date, str]
Row = Dict[str, Any]

# The columns the Scala loader writes (OptionTradeRepository.insert order), minus ingested_at (the
# ReplacingMergeTree version, filled by the CH DEFAULT on re-insert).
COLUMNS_NON_GREEKS: Tuple[str, ...] = (
    "symbol", "underlying", "session_date", "trade_ts", "price", "size", "exchange", "conditions",
    "sip_timestamp_ns", "ordinal", "sequence_number", "participant_timestamp_ns", "correction",
    "spot_at_trade", "spot_trade_ts", "quote_ts", "bid", "ask", "bid_size", "ask_size",
    "quote_lag_ms", "source",
)
COLUMNS_GREEKS: Tuple[str, ...] = GREEKS_COLUMNS  # the greeks columns the core fills
COLUMNS: Tuple[str, ...] = COLUMNS_NON_GREEKS + COLUMNS_GREEKS


def dataset_for(methodology_version: str) -> str:
    return f"greeks_{methodology_version}"


# ── pricing one (day, underlying)'s rows ─────────────────────────────────────────────────────────

def price_rows(rows: Sequence[Row], inputs: Inputs, *, methodology_version: str, t_floor_minutes: float,
               expiry_time_et: str, now: datetime) -> Tuple[List[Row], "Counter[str]"]:
    """Every row with its greeks columns filled: non-greeks columns copied EXACTLY (overlaying the
    disjoint greeks columns), and a count of failures by reason. Pure.

    All pricing lives in the shared core: this builds a TradeRow per option_trade row (symbol, price,
    spot_at_trade, trade_ts), calls greeks.trades.price_trade_rows for the whole chunk, and overlays
    the greeks columns it returns onto each row. There is no pricing logic in greeks/history/."""
    if not rows:
        return [], Counter()
    trades = [
        TradeRow(osi=row["symbol"], price=float(row["price"]),
                 spot=None if row["spot_at_trade"] is None else float(row["spot_at_trade"]),
                 as_of=row["trade_ts"])
        for row in rows
    ]
    cols, reasons = price_trade_rows(
        trades, inputs, session_date=rows[0]["session_date"],
        methodology_version=methodology_version, t_floor_minutes=t_floor_minutes,
        expiry_time_et=expiry_time_et, now=now)
    return [{**row, **col} for row, col in zip(rows, cols)], reasons


# ── orchestration ────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PairResult:
    day: date
    underlying: str
    rows: int
    priced: int                 # successes
    reasons: Mapping[str, int]  # per-row failure reasons
    seconds: float
    error: Optional[str]        # a whole-pair failure (CH/inputs); None on success


@dataclass(frozen=True)
class Summary:
    days: int
    pairs: int
    rows: int
    priced: int
    failed_pairs: int
    stopped: bool = False       # True if the run was stopped by a signal before finishing


class StopRequested(Exception):
    """Raised by the mapper after a stop signal (Ctrl-C/SIGTERM), once the pairs that were already
    running have drained (each wrote its ledger row). Carries how many pairs completed and how many
    never started, so the run can log the stop line and a rerun can continue."""
    def __init__(self, done: int, not_started: int) -> None:
        super().__init__(f"stopped: {done} done, {not_started} not started")
        self.done = done
        self.not_started = not_started


@dataclass(frozen=True)
class Deps:
    """The job's IO, injected so run() drives against fakes with no live ClickHouse and no processes.
    `process_pair` is the per-(day, underlying) unit (in production it opens its own CH client and
    reads → prices → re-inserts → ledgers); `mapper` runs it (a process pool in production, serial in
    tests)."""
    list_pairs: Callable[[date, date], List[Pair]]       # distinct (session_date, underlying) in option_trade
    ledgered: Callable[[date, date], "set[Pair]"]        # pairs with a greeks ledger row already (read ONCE)
    count_rows: Callable[[date, str], int]               # dry-run preview size
    process_pair: Callable[[Pair], PairResult]
    # Streams results AS THEY COMPLETE (production: one forkserver pool + as_completed), so a day line
    # can be logged the moment its last pair finishes. May yield out of order.
    mapper: Callable[[Callable[[Pair], PairResult], List[Pair]], Iterable[PairResult]]


@dataclass
class _DayAgg:
    """Per-day running totals so the day line can be logged on completion without holding every
    pair's result. `remaining` counts down to the day's last pair."""
    remaining: int
    ok: int = 0                                           # successfully-processed pairs (underlyings)
    rows: int = 0
    priced: int = 0
    failed: int = 0                                       # whole-pair failures (per-reason stays in the rows)
    seconds: float = 0.0


def run(from_date: date, to_date: date, dry_run: bool, deps: Deps) -> Summary:
    """Price every (day, underlying) in [from_date, to_date] not already greeks-ledgered. A dry run
    prints the pairs and their row counts and writes nothing. One INFO line per day; one summary."""
    pairs = sorted(deps.list_pairs(from_date, to_date))
    done = deps.ledgered(from_date, to_date)             # ONE ClickHouse read, kept as a set
    todo = [p for p in pairs if p not in done]

    if dry_run:
        by_day: "defaultdict[date, List[str]]" = defaultdict(list)
        for d, u in todo:
            by_day[d].append(u)
        total = 0
        for d in sorted(by_day):
            counts = [(u, deps.count_rows(d, u)) for u in sorted(by_day[d])]
            n = sum(c for _, c in counts)
            total += n
            log.info("[greeks][dry-run] %s: %d underlying(s), %d rows — %s",
                     d, len(counts), n, ", ".join(f"{u}={c}" for u, c in counts))
        log.info("[greeks][dry-run] %d pair(s) to do, %d rows — nothing written", len(todo), total)
        return Summary(days=len(by_day), pairs=len(todo), rows=total, priced=0, failed_pairs=0)

    # One executor for the whole run; consume results AS THEY COMPLETE (deps.mapper streams, possibly
    # out of order). Track only per-day running totals and the failed-pair count — never every result —
    # and log a day's INFO line the moment its last pair finishes (days may finish out of order).
    days: Dict[date, _DayAgg] = {}
    for d, _ in todo:
        agg = days.get(d)
        if agg is None:
            days[d] = _DayAgg(remaining=1)
        else:
            agg.remaining += 1

    days_done = pairs_total = rows_total = priced_total = failed_pairs = 0
    try:
        for r in deps.mapper(deps.process_pair, todo):
            pairs_total += 1
            agg = days[r.day]
            agg.remaining -= 1
            agg.seconds += r.seconds
            if r.error is not None:
                failed_pairs += 1
                agg.failed += 1
                log.warning("[greeks] %s %s FAILED — no ledger row, a rerun retries it: %s",
                            r.day, r.underlying, r.error)
            else:
                agg.ok += 1
                agg.rows += r.rows
                agg.priced += r.priced
            if agg.remaining == 0:
                failed_rows = agg.rows - agg.priced                 # rows read but not priced
                pairs_suffix = f", {agg.failed} failed pair(s)" if agg.failed else ""
                log.info("[greeks] %s: %d underlying(s), %d rows, %d priced, %d failed, %.1fs%s",
                         r.day, agg.ok, agg.rows, agg.priced, failed_rows, agg.seconds, pairs_suffix)
                days_done += 1
                rows_total += agg.rows
                priced_total += agg.priced
                del days[r.day]  # free the day's totals; nothing else is retained
    except StopRequested:
        # A clean stop: the running pairs drained (each wrote its ledger row), the rest were cancelled.
        not_started = len(todo) - pairs_total
        log.info("[greeks] stopped: %d pairs done, %d not started — rerun to continue",
                 pairs_total, not_started)
        return Summary(days=days_done, pairs=pairs_total, rows=rows_total, priced=priced_total,
                       failed_pairs=failed_pairs, stopped=True)

    log.info("[greeks] done: %d day(s), %d pair(s), %d rows, %d priced, %d failed pair(s)",
             days_done, pairs_total, rows_total, priced_total, failed_pairs)
    return Summary(days=days_done, pairs=pairs_total, rows=rows_total, priced=priced_total,
                   failed_pairs=failed_pairs)
