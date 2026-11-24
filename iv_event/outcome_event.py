"""
The ClickHouse side of method out_v1: per calendar month of session_date, read the passed windows and
every input (rv_daily, jump_daily, iv_curve_daily, corporate_action, split_effective, the market-wide
dates) over the month's SPAN, build each window's outcome with outcome.py, and insert the rows with
ONE insert. Mirrors event.py for the windows; the pure math is outcome.py (mirrors fwd.py).

The read span for a month is NOT the calendar month: a December session's window can run to February,
so every per-month read spans from the month's first session_date (and the previous trading day, for
the release's d_prev) to the LATEST end_expiry among that month's windows. A span-coverage check fails
by name and date if any window day ever falls outside the span read (it never should).

A window is passed when its end_expiry is on or before the latest date present in BOTH jump_daily
(bns_v1) and iv_curve_daily (iso_v1). outcome-backfill recomputes a date range (re-insert-safe);
`daily` is data-driven (every passed window with no out_v1 row yet, like rv_daily.incremental) — no
ledger. This package never creates or alters tables; _require_outcome_table fails fast if the table or
any column is missing.

Logging: one line per month, nothing per window.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from clickhouse_connect.driver.exceptions import OperationalError

from ml.shared.calendar import previous_trading_day, trading_days_in

from . import (
    CORPORATE_ACTION_TABLE, IV_CURVE_TABLE, IV_EVENT_OUTCOME_TABLE, IV_EVENT_TABLE, JUMP_DAILY_TABLE,
    JUMP_METHOD, MARKET_WIDE_MIN_NAMES, OUTCOME_METHOD, OUTCOME_SUMMARY_MIN_SUPPORT, RV_DAILY_TABLE,
    RV_SOURCE, SOURCE_METHOD, SPLIT_EFFECTIVE_TABLE, STATUS_OK, WINDOW_METHOD,
)
from . import event
from . import outcome as oc
from .daytype import gap_bucket
from .event import DaytypeMissing, _require_daytype

log = logging.getLogger("iv_event.outcome_event")

_UTC = ZoneInfo("UTC")

# The iv_event_outcome columns written from Python, in insert order = DDL order. Unlike iv_event_window
# (whose computed_at is a DDL DEFAULT), this DDL has no DEFAULT on computed_at, so the writer supplies
# it (now, UTC) and it is in OUTCOME_COLUMNS. The column-contract test keys on this being the full set.
OUTCOME_COLUMNS: Tuple[str, ...] = (
    "method", "underlying", "session_date", "end_td", "start_expiry", "end_expiry", "window_method",
    "n_days", "n_rv_days", "realized_var", "realized_excess_var", "n_jump_days", "first_jump_date",
    "max_abs_ret_adj", "n_release_days", "crush_date", "crush_excess_var", "corporate_action",
    "explained", "computed_at",
)

# Non-counting corporate-action kinds: a cash_dividend is subtracted at read time already and a
# name_change is not an economic event, so neither sets corporate_action (every other kind does).
_CORP_ACTION_EXCLUDED_KINDS = ("cash_dividend", "name_change")


class OutcomeTableMissing(RuntimeError):
    """trading.iv_event_outcome is absent or missing a column — the operator applies the DDL first."""


def _require_outcome_table(ch: Any) -> None:
    """Fail fast if trading.iv_event_outcome is absent or missing any column. This package never
    creates or alters tables — the operator applies the DDL in Scala's database_schema.sql first."""
    present = {r[0] for r in ch.query(
        "SELECT name FROM system.columns WHERE database = 'trading' AND table = 'iv_event_outcome'"
    ).result_rows}
    if not present:
        raise OutcomeTableMissing(
            f"{IV_EVENT_OUTCOME_TABLE} does not exist. Apply its DDL from Scala's db/database_schema.sql "
            "first; this package never creates tables.")
    missing = [c for c in OUTCOME_COLUMNS if c not in present]
    if missing:
        raise OutcomeTableMissing(
            f"{IV_EVENT_OUTCOME_TABLE} is missing column(s): " + ", ".join(missing) + ". Apply the "
            "current DDL from Scala's db/database_schema.sql (this package never creates or alters tables).")


# ── the passed boundary ────────────────────────────────────────────────────────────────────────────

def passed_max(ch: Any) -> Optional[dt.date]:
    """The latest date present in BOTH jump_daily (bns_v1) and iv_curve_daily (iso_v1) — a window is
    passed when its end_expiry is on or before this. None when either source is empty."""
    j = event._max_date(
        ch, f"SELECT max(trading_date) FROM {JUMP_DAILY_TABLE} WHERE method = %(m)s",
        {"m": JUMP_METHOD})
    c = event._max_date(
        ch, f"SELECT max(session_date) FROM {IV_CURVE_TABLE} WHERE method = %(m)s",
        {"m": SOURCE_METHOD})
    if j is None or c is None:
        return None
    return min(j, c)


# ── per-month window read ────────────────────────────────────────────────────────────────────────

def _read_windows(ch: Any, ms: dt.date, me: dt.date, pmax: dt.date,
                  only_missing: bool) -> List[oc.WindowSpec]:
    """The passed ok fwd_v1 windows with session_date in [ms, me] and end_expiry <= pmax, as
    WindowSpecs. `only_missing` drops windows that already have an out_v1 outcome row."""
    only = (f" AND (underlying, session_date, end_td) NOT IN "
            f"(SELECT underlying, session_date, end_td FROM {IV_EVENT_OUTCOME_TABLE} FINAL "
            f"WHERE method = %(om)s)"
            if only_missing else "")
    res = ch.query(
        f"SELECT underlying, session_date, start_expiry, end_expiry, end_td, eff_days, "
        f"normal_daily_var, own_earnings_date "
        f"FROM {IV_EVENT_TABLE} FINAL "
        f"WHERE method = %(m)s AND status = %(ok)s AND session_date BETWEEN %(a)s AND %(b)s "
        f"AND end_expiry IS NOT NULL AND end_expiry <= %(pmax)s{only} "
        f"ORDER BY underlying, session_date, end_td",
        parameters={"m": WINDOW_METHOD, "ok": STATUS_OK, "a": ms, "b": me, "pmax": pmax,
                    "om": OUTCOME_METHOD})
    return [oc.WindowSpec(
        underlying=str(r[0]).upper(), session_date=r[1], start_expiry=r[2], end_expiry=r[3],
        end_td=int(r[4]), eff_days=float(r[5]), normal_daily_var=float(r[6]), own_earnings_date=r[7])
        for r in res.result_rows]


# ── per-month input reads (over the SPAN, not the calendar month) ──────────────────────────────────

def _read_rv(ch: Any, underlyings: Set[str], lo: dt.date, hi: dt.date
             ) -> Dict[Tuple[str, dt.date], float]:
    if not underlyings:
        return {}
    res = ch.query(
        f"SELECT symbol, trading_date, rv_total FROM {RV_DAILY_TABLE} FINAL "
        f"WHERE source = %(src)s AND symbol IN %(u)s AND trading_date BETWEEN %(a)s AND %(b)s",
        parameters={"src": RV_SOURCE, "u": list(underlyings), "a": lo, "b": hi})
    return {(str(r[0]).upper(), r[1]): float(r[2]) for r in res.result_rows}


def _read_jumps(ch: Any, underlyings: Set[str], lo: dt.date, hi: dt.date
                ) -> Dict[Tuple[str, dt.date], oc.JumpRow]:
    if not underlyings:
        return {}
    res = ch.query(
        f"SELECT symbol, trading_date, overnight_jump, overnight_jump_adj, intraday_jump, "
        f"overnight_ret_adj FROM {JUMP_DAILY_TABLE} FINAL "
        f"WHERE method = %(m)s AND symbol IN %(u)s AND trading_date BETWEEN %(a)s AND %(b)s",
        parameters={"m": JUMP_METHOD, "u": list(underlyings), "a": lo, "b": hi})
    return {(str(r[0]).upper(), r[1]): oc.JumpRow(
        overnight_jump=None if r[2] is None else int(r[2]),
        overnight_jump_adj=None if r[3] is None else int(r[3]),
        intraday_jump=None if r[4] is None else int(r[4]),
        overnight_ret_adj=None if r[5] is None else float(r[5])) for r in res.result_rows}


def _read_curve(ch: Any, underlyings: Set[str], expiries: Set[dt.date], lo: dt.date, hi: dt.date
                ) -> Dict[Tuple[str, dt.date, dt.date], float]:
    """iv_curve_daily fit_total_var (iso_v1, kind expiry) keyed by (underlying, session_date, expiry),
    read over the window end_expiries only — the release needs TV(x, E) for those E."""
    if not underlyings or not expiries:
        return {}
    res = ch.query(
        f"SELECT underlying, session_date, expiry, fit_total_var FROM {IV_CURVE_TABLE} FINAL "
        f"WHERE method = %(m)s AND kind = 'expiry' AND underlying IN %(u)s "
        f"AND expiry IN %(e)s AND session_date BETWEEN %(a)s AND %(b)s "
        f"AND fit_total_var IS NOT NULL",
        parameters={"m": SOURCE_METHOD, "u": list(underlyings), "e": list(expiries), "a": lo, "b": hi})
    return {(str(r[0]).upper(), r[1], r[2]): float(r[3]) for r in res.result_rows}


def read_corp_actions(ch: Any, underlyings: Set[str]) -> Dict[str, List[dt.date]]:
    """Per underlying, the coalesce(ex_date, event_date) of every corporate_action row whose kind is
    not cash_dividend or name_change. Not date-filtered in SQL — each window tests its own [s, E]."""
    if not underlyings:
        return {}
    excl = ", ".join(f"'{k}'" for k in _CORP_ACTION_EXCLUDED_KINDS)
    res = ch.query(
        f"SELECT symbol, coalesce(ex_date, event_date) AS d FROM {CORPORATE_ACTION_TABLE} FINAL "
        f"WHERE symbol IN %(u)s AND kind NOT IN ({excl})",
        parameters={"u": list(underlyings)})
    out: Dict[str, List[dt.date]] = defaultdict(list)
    for sym, d in res.result_rows:
        if d is not None:
            out[str(sym).upper()].append(_as_date(d))
    return out


def read_splits(ch: Any, underlyings: Set[str]) -> Dict[str, List[dt.date]]:
    """Per underlying, every split_effective execution_date."""
    if not underlyings:
        return {}
    res = ch.query(
        f"SELECT symbol, execution_date FROM {SPLIT_EFFECTIVE_TABLE} WHERE symbol IN %(u)s",
        parameters={"u": list(underlyings)})
    out: Dict[str, List[dt.date]] = defaultdict(list)
    for sym, d in res.result_rows:
        if d is not None:
            out[str(sym).upper()].append(_as_date(d))
    return out


def read_market_wide(ch: Any, lo: dt.date, hi: dt.date) -> Set[dt.date]:
    """The dates in [lo, hi] on which at least MARKET_WIDE_MIN_NAMES names carry overnight_jump_adj=1
    in jump_daily (bns_v1) — the same count rule as the Scala market-wide definition."""
    res = ch.query(
        f"SELECT trading_date FROM {JUMP_DAILY_TABLE} FINAL "
        f"WHERE method = %(m)s AND trading_date BETWEEN %(a)s AND %(b)s "
        f"GROUP BY trading_date HAVING countIf(overnight_jump_adj = 1) >= %(n)s",
        parameters={"m": JUMP_METHOD, "a": lo, "b": hi, "n": MARKET_WIDE_MIN_NAMES})
    return {r[0] for r in res.result_rows}


def _as_date(d: Any) -> dt.date:
    """corporate_action uses Date32 (can surface as datetime); normalise to a plain date."""
    return d.date() if isinstance(d, datetime) else d


# ── the month processor ────────────────────────────────────────────────────────────────────────────

def process_outcome_month(ch: Any, ms: dt.date, me: dt.date, pmax: dt.date, weights: Dict[int, float],
                          only_missing: bool) -> int:
    """Build and insert the outcomes for every passed window with session_date in [ms, me]. Reads each
    input once over the SPAN [ms (and its previous trading day), latest end_expiry]. Returns the row
    count. Fails by name and date if any window day falls outside the span read (never should)."""
    specs = _read_windows(ch, ms, me, pmax, only_missing)
    if not specs:
        return 0

    underlyings = {s.underlying for s in specs}
    expiries = {s.end_expiry for s in specs}
    # Span: from the month's first session_date — and its previous trading day, for the release's
    # d_prev — to the latest end_expiry among this month's windows (windows can run months past me).
    span_lo = previous_trading_day(min(s.session_date for s in specs))
    span_hi = max(s.end_expiry for s in specs)

    rv = _read_rv(ch, underlyings, span_lo, span_hi)
    jumps = _read_jumps(ch, underlyings, span_lo, span_hi)
    curve = _read_curve(ch, underlyings, expiries, span_lo, span_hi)
    corp = read_corp_actions(ch, underlyings)
    splits = read_splits(ch, underlyings)
    market_wide = read_market_wide(ch, span_lo, span_hi)

    # Group the per-day reads by underlying once (not once per window).
    rv_by_u: Dict[str, Dict[dt.date, float]] = defaultdict(dict)
    for (sym, d), v in rv.items():
        rv_by_u[sym][d] = v
    jumps_by_u: Dict[str, Dict[dt.date, oc.JumpRow]] = defaultdict(dict)
    for (sym, d), jr in jumps.items():
        jumps_by_u[sym][d] = jr

    def day_weight(d: dt.date) -> float:
        g = gap_bucket(d)
        if g not in weights:
            raise ValueError(
                f"no {IV_EVENT_TABLE} daytype weight for gap {g} (needed by an outcome window day {d}) "
                "- recompute daytype over a range that includes it")
        return weights[g]

    def is_market_wide(d: dt.date) -> bool:
        return d in market_wide

    computed_at = datetime.now(tz=_UTC)
    rows: List[List[Any]] = []
    for spec in specs:
        u = spec.underlying

        def curve_tv(session: dt.date, expiry: dt.date, _u: str = u) -> Optional[float]:
            return curve.get((_u, session, expiry))

        days, out = oc.build_outcome(
            spec, list_days=trading_days_in, prev_trading_day=previous_trading_day,
            day_weight=day_weight, rv_by_date=rv_by_u.get(u, {}), jump_by_date=jumps_by_u.get(u, {}),
            curve_tv=curve_tv, ca_dates=corp.get(u, ()), split_dates=splits.get(u, ()),
            is_market_wide=is_market_wide)

        # Span-coverage check: every window day (and its d_prev) must be inside the read span, else a
        # realized/jump/release read could be silently truncated. It should never fire.
        for d in days:
            if not (span_lo <= previous_trading_day(d) and d <= span_hi):
                raise RuntimeError(
                    f"iv_event_outcome {u} {spec.session_date} end_td={spec.end_td}: window day {d} "
                    f"(d_prev {previous_trading_day(d)}) outside the read span [{span_lo}, {span_hi}]")

        rows.append([
            OUTCOME_METHOD, u, spec.session_date, spec.end_td, spec.start_expiry, spec.end_expiry,
            WINDOW_METHOD, out.n_days, out.n_rv_days, out.realized_var, out.realized_excess_var,
            out.n_jump_days, out.first_jump_date, out.max_abs_ret_adj, out.n_release_days,
            out.crush_date, out.crush_excess_var, out.corporate_action, out.explained, computed_at,
        ])

    ch.insert(IV_EVENT_OUTCOME_TABLE, rows, column_names=list(OUTCOME_COLUMNS))
    return len(rows)


# ── backfill / daily drivers ───────────────────────────────────────────────────────────────────────

def outcome_backfill(ch: Any, get_client: Callable[[], Any], start: dt.date, end: dt.date) -> int:
    """Compute outcomes for every passed window with session_date in [start, end], one calendar month
    at a time (its reads span to the windows' latest end_expiry). Re-insert-safe. Ctrl-C stops cleanly
    after the current month; a lost connection rebuilds the client once and retries that month."""
    _require_outcome_table(ch)
    pmax = passed_max(ch)
    if pmax is None:
        print("[iv_event] no passed windows yet (jump_daily or iv_curve_daily is empty)")
        return 0
    try:
        weights, _f, _t = _require_daytype(ch)
    except DaytypeMissing as exc:
        print(str(exc))
        return -1
    print(f"[iv_event] outcome-backfill {start} .. {end}  (passed through {pmax})")
    event.reset_stop()
    box: List[Any] = [ch]
    total = 0
    for ms, me in event._month_ranges(start, end):
        if event._stop_requested():
            print("[iv_event] stop requested — stopping cleanly after the current month")
            break
        t0 = time.perf_counter()
        n = _month_with_retry(box, get_client, ms, me, pmax, weights, only_missing=False)
        secs = time.perf_counter() - t0
        total += n
        print(f"[iv_event] {ms} .. {me}: {n} outcomes  ({secs:.1f}s)")
    print(f"[iv_event] outcome-backfill done: {total} outcomes")
    return total


def _month_with_retry(box: List[Any], get_client: Callable[[], Any], ms: dt.date, me: dt.date,
                      pmax: dt.date, weights: Dict[int, float], only_missing: bool) -> int:
    try:
        return process_outcome_month(box[0], ms, me, pmax, weights, only_missing)
    except OperationalError as exc:
        log.info("iv_event outcomes %s..%s: lost ClickHouse connection (%s) — rebuilding and retrying",
                 ms, me, exc)
        box[0] = get_client()
        return process_outcome_month(box[0], ms, me, pmax, weights, only_missing)


def outcome_daily(ch: Any) -> int:
    """Compute outcomes for every passed window with no out_v1 row yet (data-driven, like
    rv_daily.incremental): find the session_date range of those windows and process it month by month
    with the only-missing filter. Prints a one-line summary that deliberately does NOT match the Scala
    IvEventJob `wrote N rows` regex, so the window row-count it parses is unchanged."""
    _require_outcome_table(ch)
    pmax = passed_max(ch)
    if pmax is None:
        print("[iv_event] outcomes: no passed windows yet")
        return 0
    try:
        weights, _f, _t = _require_daytype(ch)
    except DaytypeMissing as exc:
        print(str(exc))
        return -1
    rng = ch.query(
        f"SELECT min(session_date), max(session_date) FROM {IV_EVENT_TABLE} FINAL "
        f"WHERE method = %(m)s AND status = %(ok)s AND end_expiry IS NOT NULL AND end_expiry <= %(pmax)s "
        f"AND (underlying, session_date, end_td) NOT IN "
        f"(SELECT underlying, session_date, end_td FROM {IV_EVENT_OUTCOME_TABLE} FINAL WHERE method = %(om)s)",
        parameters={"m": WINDOW_METHOD, "ok": STATUS_OK, "pmax": pmax, "om": OUTCOME_METHOD})
    lo, hi = rng.result_rows[0]
    if lo is None or lo <= dt.date(1970, 1, 1):
        print("[iv_event] outcomes: none to compute (every passed window already has a row)")
        return 0
    print(f"[iv_event] outcomes {lo} .. {hi} (every passed window with no out_v1 row)")
    total = 0
    for ms, me in event._month_ranges(lo, hi):
        total += process_outcome_month(ch, ms, me, pmax, weights, only_missing=True)
    print(f"[iv_event] outcomes: wrote {total} outcome rows")
    return total


# ── backfill summary (printed once at the end — the proof) ────────────────────────────────────────

# The join of each outcome to its iv_event_window row (for start_td / eff_days / excess_days /
# excess_var / support / normal_daily_var / own_earnings_date, which the outcome table does not carry).
# Reused by the summary parts and by the scoring report (iv_event/score.py).
def _joined_cte() -> str:
    return (
        f"o AS (SELECT * FROM {IV_EVENT_OUTCOME_TABLE} FINAL WHERE method = '{OUTCOME_METHOD}'), "
        f"w AS (SELECT underlying, session_date, end_td, start_td, eff_days, excess_days, excess_var, "
        f"             support, normal_daily_var, own_earnings_date "
        f"      FROM {IV_EVENT_TABLE} FINAL WHERE method = '{WINDOW_METHOD}' AND status = '{STATUS_OK}'), "
        f"j AS (SELECT o.*, w.start_td AS w_start_td, w.eff_days AS w_eff_days, "
        f"             w.excess_days AS w_excess_days, "
        f"             w.excess_var AS w_excess_var, w.support AS w_support, "
        f"             w.normal_daily_var AS w_normal_daily_var, "
        f"             w.own_earnings_date AS w_own_earnings_date "
        f"      FROM o INNER JOIN w "
        f"        ON o.underlying = w.underlying AND o.session_date = w.session_date "
        f"       AND o.end_td = w.end_td)")


def summary_rows_by_year(ch: Any) -> List[tuple]:
    """(year, rows, share realized_var present, share crush_date present)."""
    return ch.query(
        f"SELECT toYear(session_date) AS y, count(), "
        f"countIf(realized_var IS NOT NULL), countIf(crush_date IS NOT NULL) "
        f"FROM {IV_EVENT_OUTCOME_TABLE} FINAL WHERE method = '{OUTCOME_METHOD}' "
        f"GROUP BY y ORDER BY y"
    ).result_rows


def _breakdown(ch: Any, key_expr: str) -> List[tuple]:
    """For support >= OUTCOME_SUMMARY_MIN_SUPPORT windows grouped by `key_expr`: (key, count, share
    n_jump_days>0, share crush_excess_var>0, median realized_excess_var/excess_var) where that ratio
    is over rows with a non-NULL realized_excess_var and a non-zero excess_var. count >= 100 to show."""
    return ch.query(
        f"WITH {_joined_cte()} "
        f"SELECT {key_expr} AS k, count() AS n, "
        f"countIf(n_jump_days > 0) / n AS jump_share, "
        f"countIf(crush_excess_var > 0) / n AS crush_share, "
        f"medianIf(realized_excess_var / w_excess_var, "
        f"         realized_excess_var IS NOT NULL AND w_excess_var != 0) AS med_ratio "
        f"FROM j WHERE w_support >= {OUTCOME_SUMMARY_MIN_SUPPORT!r} "
        f"GROUP BY k HAVING n >= 100 ORDER BY k"
    ).result_rows


def summary_by_explained(ch: Any) -> List[tuple]:
    return _breakdown(ch, "explained")


def summary_by_excess_days_band(ch: Any) -> List[tuple]:
    band = ("multiIf(w_excess_days < 1, '<1', w_excess_days < 2, '1-2', "
            "w_excess_days < 5, '2-5', '5+')")
    return _breakdown(ch, band)


def summary_crush_day_agreement(ch: Any) -> tuple:
    """(n, agree) — own_earnings windows with a crush_date, and how many have crush_date equal to the
    window's own_earnings_date."""
    row = ch.query(
        f"WITH {_joined_cte()} "
        f"SELECT count(), countIf(crush_date = w_own_earnings_date) "
        f"FROM j WHERE explained = '{oc.EXPLAINED_OWN_EARNINGS}' AND crush_date IS NOT NULL"
    ).result_rows[0]
    return (int(row[0]), int(row[1]))


def summary_corp_action_by_year(ch: Any) -> List[tuple]:
    return ch.query(
        f"SELECT toYear(session_date) AS y, countIf(corporate_action = 1) "
        f"FROM {IV_EVENT_OUTCOME_TABLE} FINAL WHERE method = '{OUTCOME_METHOD}' "
        f"GROUP BY y ORDER BY y"
    ).result_rows


# The named known cases: (underlying, session_date, note). Printed with all outcome columns.
_KNOWN_CASES = (
    ("EWZ", dt.date(2026, 10, 23), "Stage 2 EWZ event"),
    ("SIRI", dt.date(2024, 7, 24), "Liberty Sirius XM merger (closed Sep 2024) → corporate_action=1 if in window"),
    ("LYFT", dt.date(2024, 2, 9), "window containing 2024-02-14 (earnings reaction)"),
    ("SPY", dt.date(2024, 8, 2), "window containing 2024-08-05 → expect a jump, explained market_wide"),
)


def summary_known_case(ch: Any, underlying: str, session_date: dt.date) -> List[tuple]:
    return ch.query(
        f"SELECT end_td, start_expiry, end_expiry, n_days, n_rv_days, realized_var, "
        f"realized_excess_var, n_jump_days, first_jump_date, max_abs_ret_adj, n_release_days, "
        f"crush_date, crush_excess_var, corporate_action, explained "
        f"FROM {IV_EVENT_OUTCOME_TABLE} FINAL "
        f"WHERE method = %(m)s AND underlying = %(u)s AND session_date = %(d)s ORDER BY end_td",
        parameters={"m": OUTCOME_METHOD, "u": underlying.upper(), "d": session_date}).result_rows


def _f(x: Any, fmt: str = "{:.3e}") -> str:
    return "—" if x is None else fmt.format(float(x))


def print_backfill_summary(ch: Any) -> None:
    print("\n[iv_event] ── outcome backfill summary ──")

    print("[iv_event] 1. rows by year (share realized_var present / share crush_date present):")
    print(f"    {'year':6s} {'rows':>10s} {'realized':>12s} {'crush':>12s}")
    for y, n, nreal, ncrush in summary_rows_by_year(ch):
        rs = f"{(nreal / n * 100 if n else 0):.1f}%"
        cs = f"{(ncrush / n * 100 if n else 0):.1f}%"
        print(f"    {y:<6d} {n:>10,d} {rs:>12s} {cs:>12s}")

    print("[iv_event] 2. by explained and by excess_days band (support >= "
          f"{OUTCOME_SUMMARY_MIN_SUPPORT:g}, groups with >= 100):")
    print(f"    {'group':8s} {'n':>8s} {'jump>0':>9s} {'crush>0':>9s} {'med rxs/xs':>12s}")
    for label, rows in (("explained", summary_by_explained(ch)),
                        ("band", summary_by_excess_days_band(ch))):
        print(f"    — by {label}")
        for k, n, jsh, csh, med in rows:
            print(f"    {str(k):8s} {int(n):>8,d} {jsh * 100:>8.1f}% {csh * 100:>8.1f}% "
                  f"{_f(med, '{:.3f}'):>12s}")

    n, agree = summary_crush_day_agreement(ch)
    share = (agree / n * 100) if n else 0.0
    print(f"[iv_event] 3. crush-day agreement (own_earnings windows with a crush_date): "
          f"{agree}/{n} = {share:.1f}% have crush_date = own_earnings_date")

    print("[iv_event] 4. known cases (all outcome columns):")
    for u, d, note in _KNOWN_CASES:
        print(f"    {u} {d} — {note}")
        rows = summary_known_case(ch, u, d)
        if not rows:
            print("        (no passed-window outcome rows)")
            continue
        for (end_td, se, ee, nd, nrv, rv, rxv, njd, fjd, mar, nrd, cd, cxv, corp, expl) in rows:
            start_s = str(se) if se is not None else "anchor"
            print(f"        end_td={int(end_td):>3d} {start_s:>10s}->{str(ee):10s} "
                  f"n_days={int(nd):>2d} n_rv={int(nrv):>2d} rv={_f(rv)} rxs={_f(rxv)} "
                  f"n_jump={int(njd)} first_jump={fjd if fjd else '—'} "
                  f"max|ra|={_f(mar)} n_rel={int(nrd)} crush={cd if cd else '—'} "
                  f"crush_xs={_f(cxv)} corp={int(corp)} expl={expl}")

    print("[iv_event] 5. corporate_action = 1 count by year:")
    print(f"    {'year':6s} {'corp=1':>10s}")
    for y, n in summary_corp_action_by_year(ch):
        print(f"    {y:<6d} {int(n):>10,d}")
