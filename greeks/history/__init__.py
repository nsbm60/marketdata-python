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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from greeks.domain import Dividend, FailureReason, OptionRight, RowStatus
from greeks.forwards.carry import carry
from greeks.occ import parse_occ
from greeks.pricing import price_option
from greeks.solver.black76 import greeks_batch
from greeks.solver.invert import invert_batch

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


def _blank_greeks(methodology_version: str, now: datetime) -> Dict[str, Any]:
    g: Dict[str, Any] = {c: None for c in COLUMNS_GREEKS}
    g["greeks_methodology"] = methodology_version
    g["greeks_basis"] = BASIS_TRADE
    g["greeks_computed_at"] = now
    return g


def _fail_greeks(g: Dict[str, Any], reason: FailureReason) -> Dict[str, Any]:
    g["greeks_status"] = RowStatus.FAILURE.value
    g["greeks_reason"] = reason.value
    return g


def _greeks_for(row: Row, inputs: Inputs, methodology_version: str, t_floor_minutes: float,
                expiry_time_et: str, now: datetime) -> Dict[str, Any]:
    """Scalar reference: one trade row priced through the shared core (greeks.pricing.price_option).
    The batch ``price_rows`` must match this row for row (pinned by a test). A non-standard OSI, an
    absent spot, or a solver reason is a failure with the greeks left None — nothing defaulted."""
    g = _blank_greeks(methodology_version, now)
    try:
        c = parse_occ(row["symbol"])
    except ValueError:
        return _fail_greeks(g, FailureReason.NONSTANDARD_CONTRACT)
    spot = row["spot_at_trade"]
    if spot is None:
        return _fail_greeks(g, FailureReason.MISSING_SPOT)
    try:
        p = price_option(
            expiry=c.expiry, right=c.right, strike=c.strike, price=float(row["price"]),
            spot=float(spot), as_of=_as_utc(row["trade_ts"]), session_date=row["session_date"],
            rate_for=inputs.rate_for, dividends=inputs.dividends_for(c.root),
            methodology_version=methodology_version, t_floor_minutes=t_floor_minutes,
            expiry_time_et=expiry_time_et)
    except Exception as e:  # noqa: BLE001 — per-row isolation, like compute_or_fail
        log.debug("[greeks] %s: price_option raised %s: %s — solver_failed", row["symbol"], type(e).__name__, e)
        return _fail_greeks(g, FailureReason.SOLVER_FAILED)
    g["greeks_status"] = p.status.value
    g["greeks_reason"] = None if p.reason is None else p.reason.value
    g["greeks_forward"], g["greeks_discount"], g["greeks_tte"] = p.forward, p.discount, p.tte
    g["iv"], g["delta"], g["gamma"] = p.iv, p.delta, p.gamma
    g["theta"], g["vega"], g["rho"] = p.theta, p.vega, p.rho
    return g


def price_rows(rows: Sequence[Row], inputs: Inputs, *, methodology_version: str, t_floor_minutes: float,
               expiry_time_et: str, now: datetime) -> Tuple[List[Row], "Counter[str]"]:
    """Every row with its greeks columns filled: non-greeks columns copied EXACTLY (overlaying the
    disjoint greeks columns), and a count of failures by reason. Pure.

    One pass does the per-row carry through the shared carry (its instants memoized by the previous
    set); the trade-price inversion and the greeks then run as a single batch over the whole chunk —
    one native IV call and one vectorized greeks pass — through the same core as the live server's
    scalar path. Results are identical to _greeks_for row for row."""
    out: List[Optional[Row]] = [None] * len(rows)
    reasons: "Counter[str]" = Counter()

    # Rows that reach the solver, staged into parallel arrays for one batch call.
    prices: List[float] = []
    strikes: List[float] = []
    calls: List[bool] = []
    forwards: List[float] = []
    discounts: List[float] = []
    ttes: List[float] = []
    spots: List[float] = []
    staged: List[int] = []

    for i, row in enumerate(rows):
        g = _blank_greeks(methodology_version, now)
        try:
            c = parse_occ(row["symbol"])  # memoized
        except ValueError:
            out[i] = {**row, **_fail_greeks(g, FailureReason.NONSTANDARD_CONTRACT)}
            reasons[FailureReason.NONSTANDARD_CONTRACT.value] += 1
            continue
        spot = row["spot_at_trade"]
        if spot is None:
            out[i] = {**row, **_fail_greeks(g, FailureReason.MISSING_SPOT)}
            reasons[FailureReason.MISSING_SPOT.value] += 1
            continue
        if c.expiry < row["session_date"]:  # expired before the session: no carry, like price_option
            out[i] = {**row, **_fail_greeks(g, FailureReason.T_BELOW_FLOOR)}
            reasons[FailureReason.T_BELOW_FLOOR.value] += 1
            continue
        try:
            cr = carry(float(spot), inputs.rate_for(c.expiry), _as_utc(row["trade_ts"]), c.expiry,
                       inputs.dividends_for(c.root), expiry_time_et=expiry_time_et)
        except Exception as e:  # noqa: BLE001 — per-row isolation, like compute_or_fail
            log.debug("[greeks] %s: carry raised %s: %s — solver_failed", row["symbol"], type(e).__name__, e)
            out[i] = {**row, **_fail_greeks(g, FailureReason.SOLVER_FAILED)}
            reasons[FailureReason.SOLVER_FAILED.value] += 1
            continue
        prices.append(float(row["price"]))
        strikes.append(c.strike)
        calls.append(c.right is OptionRight.CALL)
        forwards.append(cr.forward)
        discounts.append(cr.discount)
        ttes.append(cr.time_to_expiry)
        spots.append(float(spot))
        staged.append(i)

    if staged:
        price = np.array(prices, dtype=np.float64)
        strike = np.array(strikes, dtype=np.float64)
        is_call = np.array(calls, dtype=bool)
        fwd = np.array(forwards, dtype=np.float64)
        disc = np.array(discounts, dtype=np.float64)
        tte = np.array(ttes, dtype=np.float64)
        spot_arr = np.array(spots, dtype=np.float64)
        iv, status, reason = invert_batch(
            price, strike, is_call, fwd, disc, tte,
            methodology_version=methodology_version, t_floor_minutes=t_floor_minutes)
        # greeks for every staged row at its IV (failed rows get a dummy sigma; their greeks are unused).
        sigma = np.where(np.isfinite(iv) & (iv > 0), iv, 1.0)
        bg = greeks_batch(fwd, strike, tte, disc, sigma, is_call, spot_arr)
        for j, i in enumerate(staged):
            g = _blank_greeks(methodology_version, now)
            g["greeks_forward"], g["greeks_discount"], g["greeks_tte"] = forwards[j], discounts[j], ttes[j]
            if status[j] is RowStatus.SUCCESS:
                g["greeks_status"] = RowStatus.SUCCESS.value
                g["iv"] = float(iv[j])
                g["delta"], g["gamma"] = float(bg.delta[j]), float(bg.gamma[j])
                g["theta"], g["vega"], g["rho"] = float(bg.theta[j]), float(bg.vega[j]), float(bg.rho[j])
            else:
                g["greeks_status"] = RowStatus.FAILURE.value
                g["greeks_reason"] = reason[j].value
                reasons[reason[j].value] += 1
            out[i] = {**rows[i], **g}

    return [r for r in out if r is not None], reasons


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
