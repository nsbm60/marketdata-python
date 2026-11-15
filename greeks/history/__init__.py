"""Historical greeks on every option_trade row (trade basis), written back into option_trade.

Every row's IV + greeks come from the ONE shared core (greeks.pricing.price_option): the trade price
inverted at trade_ts against spot_at_trade (the respot corrected it). Live greeks price the same way
off a quote mid — only the basis differs, recorded per row (greeks_basis='trade' here).

The rewrite follows RespotJob, per (day, underlying): read the day's rows (FINAL), load the day's
inputs once, price, re-insert the FULL rows with the greeks columns filled and a fresh ingested_at
(ReplacingMergeTree(ingested_at) keeps the new version) — every other column copied exactly — then
write an option_archive_ledger row with dataset='greeks_<methodology>'. A rerun skips pairs that
already have that ledger row, so the job is resumable and safe to re-run. Workers are independent
processes (one (day, underlying) each) — no asyncio/threading under greeks/.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from greeks.domain import Dividend, FailureReason, RowStatus
from greeks.occ import parse_occ
from greeks.pricing import price_option

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
COLUMNS_GREEKS: Tuple[str, ...] = (
    "greeks_methodology", "greeks_basis", "greeks_status", "greeks_reason",
    "greeks_forward", "greeks_discount", "greeks_tte", "iv", "delta", "gamma", "theta", "vega", "rho",
    "greeks_computed_at",
)
COLUMNS: Tuple[str, ...] = COLUMNS_NON_GREEKS + COLUMNS_GREEKS

BASIS_TRADE = "trade"


def dataset_for(methodology_version: str) -> str:
    return f"greeks_{methodology_version}"


# ── pricing one (day, underlying)'s rows ─────────────────────────────────────────────────────────

# The per-day inputs this job needs: the SOFR→rate lookup and the dividends for an underlying. Live
# code's SessionInputs satisfies this structurally (its rate_for / dividends_for), so the pure pricer
# stays independent of greeks_server/ — only the production wiring loads a concrete SessionInputs.
class Inputs(Protocol):
    def rate_for(self, expiry: date) -> float: ...
    def dividends_for(self, underlying: str) -> Sequence[Dividend]: ...


def _as_utc(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _greeks_for(row: Row, inputs: Inputs, methodology_version: str, t_floor_minutes: float,
                expiry_time_et: str, now: datetime) -> Dict[str, Any]:
    """The greeks columns for one trade row: the trade price inverted against spot_at_trade. A
    non-standard OSI, an absent spot, or a solver reason is a failure with the greeks left None —
    nothing defaulted. forward/discount/tte are filled once the carry was computed."""
    g: Dict[str, Any] = {c: None for c in COLUMNS_GREEKS}
    g["greeks_methodology"] = methodology_version
    g["greeks_basis"] = BASIS_TRADE
    g["greeks_computed_at"] = now

    def fail(reason: FailureReason) -> Dict[str, Any]:
        g["greeks_status"] = RowStatus.FAILURE.value
        g["greeks_reason"] = reason.value
        return g

    try:
        c = parse_occ(row["symbol"])
    except ValueError:
        return fail(FailureReason.NONSTANDARD_CONTRACT)
    spot = row["spot_at_trade"]
    if spot is None:
        return fail(FailureReason.MISSING_SPOT)
    try:
        p = price_option(
            expiry=c.expiry, right=c.right, strike=c.strike, price=float(row["price"]),
            spot=float(spot), as_of=_as_utc(row["trade_ts"]), session_date=row["session_date"],
            rate_for=inputs.rate_for, dividends=inputs.dividends_for(c.root),
            methodology_version=methodology_version, t_floor_minutes=t_floor_minutes,
            expiry_time_et=expiry_time_et)
    except Exception as e:  # noqa: BLE001 — per-row isolation, like compute_or_fail
        log.debug("[greeks] %s: price_option raised %s: %s — solver_failed", row["symbol"], type(e).__name__, e)
        return fail(FailureReason.SOLVER_FAILED)

    g["greeks_status"] = p.status.value
    g["greeks_reason"] = None if p.reason is None else p.reason.value
    g["greeks_forward"], g["greeks_discount"], g["greeks_tte"] = p.forward, p.discount, p.tte
    g["iv"], g["delta"], g["gamma"] = p.iv, p.delta, p.gamma
    g["theta"], g["vega"], g["rho"] = p.theta, p.vega, p.rho
    return g


def price_rows(rows: Sequence[Row], inputs: Inputs, *, methodology_version: str, t_floor_minutes: float,
               expiry_time_et: str, now: datetime) -> Tuple[List[Row], "Counter[str]"]:
    """Every row with its greeks columns filled: non-greeks columns copied EXACTLY (overlaying the
    disjoint greeks columns), and a count of failures by reason. Pure."""
    out: List[Row] = []
    reasons: "Counter[str]" = Counter()
    for r in rows:
        g = _greeks_for(r, inputs, methodology_version, t_floor_minutes, expiry_time_et, now)
        out.append({**r, **g})  # disjoint key sets: non-greeks + greeks
        if g["greeks_status"] == RowStatus.FAILURE.value:
            reasons[str(g["greeks_reason"])] += 1
    return out, reasons


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


@dataclass(frozen=True)
class Deps:
    """The job's IO, injected so run() drives against fakes with no live ClickHouse and no processes.
    `process_pair` is the per-(day, underlying) unit (in production it opens its own CH client and
    reads → prices → re-inserts → ledgers); `mapper` runs it (a process pool in production, serial in
    tests)."""
    list_pairs: Callable[[date, date], List[Pair]]       # distinct (session_date, underlying) in option_trade
    ledgered: Callable[[date, date], "set[Pair]"]        # pairs with a greeks ledger row already
    count_rows: Callable[[date, str], int]               # dry-run preview size
    process_pair: Callable[[Pair], PairResult]
    mapper: Callable[[Callable[[Pair], PairResult], List[Pair]], List[PairResult]]


def run(from_date: date, to_date: date, dry_run: bool, deps: Deps) -> Summary:
    """Price every (day, underlying) in [from_date, to_date] not already greeks-ledgered. A dry run
    prints the pairs and their row counts and writes nothing. One INFO line per day; one summary."""
    pairs = sorted(deps.list_pairs(from_date, to_date))
    todo = [p for p in pairs if p not in deps.ledgered(from_date, to_date)]

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

    results = deps.mapper(deps.process_pair, todo)
    by_day_res: "defaultdict[date, List[PairResult]]" = defaultdict(list)
    for r in results:
        by_day_res[r.day].append(r)

    rows_total = priced_total = failed_pairs = 0
    for d in sorted(by_day_res):
        rs = by_day_res[d]
        ok = [r for r in rs if r.error is None]
        reasons: "Counter[str]" = Counter()
        for r in rs:
            if r.error is not None:
                failed_pairs += 1
                log.warning("[greeks] %s %s FAILED — no ledger row, a rerun retries it: %s",
                            d, r.underlying, r.error)
            else:
                reasons.update(r.reasons)
        day_rows = sum(r.rows for r in ok)
        day_priced = sum(r.priced for r in ok)
        rows_total += day_rows
        priced_total += day_priced
        log.info("[greeks] %s: %d underlying(s), %d rows, %d priced, failed=%s, %.1fs",
                 d, len(ok), day_rows, day_priced, dict(reasons) or "none", sum(r.seconds for r in rs))

    log.info("[greeks] done: %d day(s), %d pair(s), %d rows, %d priced, %d failed pair(s)",
             len(by_day_res), len(results), rows_total, priced_total, failed_pairs)
    return Summary(days=len(by_day_res), pairs=len(results), rows=rows_total, priced=priced_total,
                   failed_pairs=failed_pairs)
