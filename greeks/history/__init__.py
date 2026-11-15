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
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from greeks.domain import Dividend, FailureReason, OptionRight, RowStatus, SolverInput
from greeks.forwards.carry import ex_date_instant_utc
from greeks.occ import parse_occ
from greeks.pricing import PricedGreeks, price_option
from greeks.solver.invert import invert
from greeks.solver.time import expiry_instant_utc

# ACT/365 seconds per year, matching greeks.solver.time / greeks.forwards.carry (kept identical).
_SECONDS_PER_ACT365_YEAR = 365.0 * 86400.0

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


def _fill_from_priced(g: Dict[str, Any], p: PricedGreeks) -> Dict[str, Any]:
    g["greeks_status"] = p.status.value
    g["greeks_reason"] = None if p.reason is None else p.reason.value
    g["greeks_forward"], g["greeks_discount"], g["greeks_tte"] = p.forward, p.discount, p.tte
    g["iv"], g["delta"], g["gamma"] = p.iv, p.delta, p.gamma
    g["theta"], g["vega"], g["rho"] = p.theta, p.vega, p.rho
    return g


def _greeks_for(row: Row, inputs: Inputs, methodology_version: str, t_floor_minutes: float,
                expiry_time_et: str, now: datetime) -> Dict[str, Any]:
    """Reference (uncached) per-row pricing: the trade price inverted against spot_at_trade through
    the shared core. A non-standard OSI, an absent spot, or a solver reason is a failure with the
    greeks left None — nothing defaulted. The cached fast path (``price_rows``) must match this row
    for row — ``test_cached_path_matches_the_reference_path`` pins it."""
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
    return _fill_from_priced(g, p)


# ── cached fast path: the same math, the per-contract and per-(underlying,day) work done once ──────

@dataclass(frozen=True)
class _Contract:
    """A symbol parsed once per pair, with its expiry instant precomputed (parse_occ and
    expiry_instant_utc are otherwise per-row costs)."""
    root: str
    expiry: date
    right: OptionRight
    strike: float
    expiry_instant: datetime
    expired_before_session: bool


def _build_contract(symbol: str, session_date: date, expiry_time_et: str) -> Optional[_Contract]:
    try:
        c = parse_occ(symbol)
    except ValueError:
        return None  # NONSTANDARD_CONTRACT, decided once per distinct symbol
    return _Contract(
        root=c.root, expiry=c.expiry, right=c.right, strike=c.strike,
        expiry_instant=expiry_instant_utc(c.expiry, expiry_time_et),
        expired_before_session=c.expiry < session_date)


def _years(instant: datetime, as_of: datetime) -> float:
    """ACT/365 years from ``as_of`` to ``instant`` — identical to greeks.solver.time.years_to_expiry
    / years_to_ex_date (both aware, so the subtraction is timezone-independent)."""
    seconds = (instant - as_of).total_seconds()
    return seconds / _SECONDS_PER_ACT365_YEAR if seconds > 0 else 0.0


def _price_cached(contract: _Contract, rate: float, dividends: Sequence[Tuple[Dividend, datetime]],
                  price: float, spot: float, as_of: datetime, methodology_version: str,
                  t_floor_minutes: float) -> PricedGreeks:
    """price_option for a pre-parsed contract with a cached rate and cached dividend ex-date instants.
    Mirrors greeks.pricing.price_option + greeks.forwards.carry exactly, computing only the trade-time-
    dependent terms (T, D, F) per row."""
    if contract.expired_before_session:
        return PricedGreeks(RowStatus.FAILURE, FailureReason.T_BELOW_FLOOR,
                            None, None, None, None, None, None, None, None, None)
    # carry(): T, then D = exp(-rT), then escrowed forward F (each dividend future-valued from its own
    # ex-date, only those strictly inside the remaining life).
    t = _years(contract.expiry_instant, as_of)
    d = 1.0 if t <= 0 else math.exp(-rate * t)
    if spot <= 0:
        raise ValueError("spot must be > 0")
    if t == 0:
        f = spot
    else:
        f = spot * math.exp(rate * t)
        for div, ex_instant in dividends:
            if div.amount < 0:
                raise ValueError(f"dividend amount must be >= 0, got {div.amount}")
            t_i = _years(ex_instant, as_of)
            if t_i <= 0.0 or t_i >= t:
                continue
            f -= div.amount * math.exp(rate * (t - t_i))
    res = invert(
        SolverInput(trade_price=price, strike=contract.strike, right=contract.right,
                    forward=f, discount=d, time_to_expiry=t, spot=spot),
        methodology_version=methodology_version, t_floor_minutes=t_floor_minutes)
    if res.status is not RowStatus.SUCCESS:
        assert res.reason_code is not None
        return PricedGreeks(RowStatus.FAILURE, res.reason_code,
                            None, None, None, None, None, None, f, d, t)
    return PricedGreeks(
        status=RowStatus.SUCCESS, reason=None,
        iv=res.iv, delta=res.delta, gamma=res.gamma, theta=res.theta, vega=res.vega, rho=res.rho,
        forward=f, discount=d, tte=t, theta_limited=res.theta_interpretively_limited)


def _greeks_for_cached(row: Row, contract: Optional[_Contract], rate: float,
                       dividends: Sequence[Tuple[Dividend, datetime]], methodology_version: str,
                       t_floor_minutes: float, now: datetime) -> Dict[str, Any]:
    """The cached counterpart of _greeks_for — same failures, same order, same math."""
    g = _blank_greeks(methodology_version, now)
    if contract is None:
        return _fail_greeks(g, FailureReason.NONSTANDARD_CONTRACT)
    spot = row["spot_at_trade"]
    if spot is None:
        return _fail_greeks(g, FailureReason.MISSING_SPOT)
    try:
        p = _price_cached(contract, rate, dividends, float(row["price"]), float(spot),
                          _as_utc(row["trade_ts"]), methodology_version, t_floor_minutes)
    except Exception as e:  # noqa: BLE001 — per-row isolation, like _greeks_for
        log.debug("[greeks] %s: cached price raised %s: %s — solver_failed", row["symbol"], type(e).__name__, e)
        return _fail_greeks(g, FailureReason.SOLVER_FAILED)
    return _fill_from_priced(g, p)


def price_rows(rows: Sequence[Row], inputs: Inputs, *, methodology_version: str, t_floor_minutes: float,
               expiry_time_et: str, now: datetime) -> Tuple[List[Row], "Counter[str]"]:
    """Every row with its greeks columns filled: non-greeks columns copied EXACTLY (overlaying the
    disjoint greeks columns), and a count of failures by reason. Pure.

    Per pair: each distinct symbol is parsed once (with its expiry instant), each distinct expiry's
    rate is looked up once, and each underlying's dividend ex-date instants are computed once; a row
    then evaluates only what depends on its own trade time. Results are identical to _greeks_for."""
    contracts: Dict[str, Optional[_Contract]] = {}
    rates: Dict[date, float] = {}
    div_instants: Dict[str, List[Tuple[Dividend, datetime]]] = {}
    out: List[Row] = []
    reasons: "Counter[str]" = Counter()
    for row in rows:
        symbol = str(row["symbol"])
        if symbol in contracts:
            contract = contracts[symbol]
        else:
            contract = _build_contract(symbol, row["session_date"], expiry_time_et)
            contracts[symbol] = contract
        # Only a priceable (parseable, not-yet-expired) contract needs a rate and dividends — and only
        # then is rate_for called, exactly as price_option would (an expired contract never looks one up).
        if contract is not None and not contract.expired_before_session:
            rate = rates.get(contract.expiry)
            if rate is None:
                rate = inputs.rate_for(contract.expiry)
                rates[contract.expiry] = rate
            divs = div_instants.get(contract.root)
            if divs is None:
                divs = [(d, ex_date_instant_utc(d.ex_date)) for d in inputs.dividends_for(contract.root)]
                div_instants[contract.root] = divs
        else:
            rate, divs = 0.0, []  # unused: contract is None or expired → fails before carry
        g = _greeks_for_cached(row, contract, rate, divs, methodology_version, t_floor_minutes, now)
        out.append({**row, **g})  # disjoint key sets: non-greeks + greeks
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
