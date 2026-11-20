"""
The ClickHouse side of method fwd_v1: the month/day aggregation over iv_curve_daily, the per-pair
window builds, the single insert into iv_event_window, the ledger, and the backfill/daily drivers.

Modelled on iv_curve/curve.py: one aggregation query per call collapses a date range to the ok iso_v1
expiry points of each (underlying, session_date); fwd.py builds the windows; the rows go in with one
insert. The ledger (dataset iv_event_fwd_v1) carries one row per (session_date, underlying) PRESENT in
iv_curve_daily for the range — including pairs with no windows (rows_inserted = 0) — so a restart
tells a fully-processed range from a partial one by a subset test.

The day weights (trading.iv_event_daytype) and each underlying's earnings are read ONCE per call, not
per pair. A day whose gap has no daytype weight is an error naming the gap and the day — no defaulting.

Logging: the per-month/day summary is PRINTED once (not also logged); a reconnect is one log.info;
nothing per pair at INFO.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from clickhouse_connect.driver.exceptions import OperationalError

from ml.shared.calendar import (
    is_trading_day, next_trading_day, previous_trading_day, session_close_et, trading_days_in,
)

from . import (
    CURVE_KIND_EXPIRY, CURVE_STATUS_OK, EARNINGS_CALENDAR, EARNINGS_CALENDAR_TABLE, EARNINGS_HISTORY,
    EARNINGS_HISTORY_TABLE, GAP_MAX_BUCKET, IV_CURVE_TABLE, IV_EVENT_DAYTYPE_TABLE, IV_EVENT_TABLE,
    LEDGER_DATASET, LEDGER_TABLE, METHOD, SOURCE_METHOD, STATUS_INSUFFICIENT_WINDOWS, STATUS_OK,
)
from .fwd import (
    ExpiryNode, Window, build_day, calendar_reaction_session, earliest_in_window,
    history_reaction_session, select_nodes,
)

log = logging.getLogger("iv_event.event")

Pair = Tuple[dt.date, str]
_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

# The iv_event_window columns written from Python, in insert order. computed_at is the
# ReplacingMergeTree version, left to the DDL DEFAULT (now64(3)). The column-contract test keys on this.
COLUMNS: Tuple[str, ...] = (
    "method", "underlying", "session_date", "start_expiry", "end_expiry", "start_td", "end_td",
    "eff_days", "window_var", "normal_daily_var", "excess_var", "excess_days", "implied_move",
    "support", "own_earnings_date", "own_earnings_source", "status", "source_method",
    "daytype_from_date", "daytype_to_date",
)

_LEDGER_COLUMNS = ["session_date", "ticker", "rows_inserted", "quote_failures", "dataset",
                   "gap_reason", "claimed_by"]

# Status display order, for the one-line summary.
_STATUS_ORDER = (STATUS_OK, STATUS_INSUFFICIENT_WINDOWS)

# Clean-stop flag: __main__ installs a SIGINT handler that sets it; backfill polls it at each month
# boundary and stops after the current month (a plain flag set from a signal handler — no threading).
_STOP: List[bool] = [False]


class DaytypeMissing(Exception):
    """No trading.iv_event_daytype rows for this method — the backfill/daily can't weight days."""


def request_stop() -> None:
    _STOP[0] = True


def reset_stop() -> None:
    _STOP[0] = False


def _stop_requested() -> bool:
    return _STOP[0]


# ── day weights ──────────────────────────────────────────────────────────────────────────────────

def _load_daytype(ch: Any) -> Optional[Tuple[Dict[int, float], dt.date, dt.date]]:
    res = ch.query(
        f"SELECT gap_days, weight, from_date, to_date FROM {IV_EVENT_DAYTYPE_TABLE} FINAL "
        "WHERE method = %(m)s ORDER BY gap_days",
        parameters={"m": METHOD})
    rows = res.result_rows
    if not rows:
        return None
    weights = {int(r[0]): float(r[1]) for r in rows}
    return weights, rows[0][2], rows[0][3]


def _require_daytype(ch: Any) -> Tuple[Dict[int, float], dt.date, dt.date]:
    loaded = _load_daytype(ch)
    if loaded is None:
        raise DaytypeMissing(
            f"no {IV_EVENT_DAYTYPE_TABLE} rows for method {METHOD} - "
            "run `python -m iv_event daytype <from> <to>` first")
    return loaded


def _gap_bucket(d: dt.date) -> int:
    return min((d - previous_trading_day(d)).days, GAP_MAX_BUCKET)


def _day_weight_fn(weights: Mapping[int, float], underlying: str,
                   session_date: dt.date) -> Callable[[dt.date], float]:
    """A weight-of-a-day function that errors (naming the gap and the day) on a gap with no daytype
    row — no defaulting, as every other missing input in this pipeline."""
    def day_weight(d: dt.date) -> float:
        g = _gap_bucket(d)
        if g not in weights:
            raise ValueError(
                f"no {IV_EVENT_DAYTYPE_TABLE} weight for gap {g} (needed by {underlying} "
                f"{session_date}, day {d}) - recompute daytype over a range that includes it")
        return weights[g]
    return day_weight


def _own_earnings_fn(candidates: List[Tuple[dt.date, str]]
                     ) -> Callable[[dt.date, dt.date], Optional[Tuple[dt.date, str]]]:
    def own(start_exclusive: dt.date, end_inclusive: dt.date) -> Optional[Tuple[dt.date, str]]:
        return earliest_in_window(candidates, start_exclusive, end_inclusive)
    return own


# ── earnings ─────────────────────────────────────────────────────────────────────────────────────

def _to_et(ts: datetime) -> datetime:
    """A UTC earnings_timestamp as a naive America/New_York datetime, for the session-close compare."""
    aware = ts if ts.tzinfo is not None else ts.replace(tzinfo=_UTC)
    return aware.astimezone(_ET).replace(tzinfo=None)


def _load_earnings(ch: Any, underlyings: Set[str],
                   today: dt.date) -> Dict[str, List[Tuple[dt.date, str]]]:
    """Each underlying's earnings reaction sessions: from earnings_history for past reports (reaction
    day, else reported day), and from earnings_calendar for future reports (the session that reacts to
    the report time). Read once; attributed to windows in Python."""
    out: Dict[str, List[Tuple[dt.date, str]]] = defaultdict(list)
    if not underlyings:
        return out

    hres = ch.query(
        f"SELECT symbol, reported_date, reaction_date FROM {EARNINGS_HISTORY_TABLE} FINAL")
    for symbol, reported, reaction in hres.result_rows:
        sym = str(symbol).upper()
        if sym not in underlyings:
            continue
        rs = history_reaction_session(reported, reaction)
        if rs <= today:   # history is the source for past reports
            out[sym].append((rs, EARNINGS_HISTORY))

    cres = ch.query(
        f"SELECT symbol, earnings_date, earnings_timestamp FROM {EARNINGS_CALENDAR_TABLE} FINAL")
    for symbol, eday, ets in cres.result_rows:
        sym = str(symbol).upper()
        if sym not in underlyings or eday <= today:   # calendar is the source for future reports
            continue
        et = _to_et(ets) if ets is not None else None
        rs = calendar_reaction_session(eday, et, session_close_et=session_close_et,
                                       is_trading_day=is_trading_day, next_trading_day=next_trading_day)
        out[sym].append((rs, EARNINGS_CALENDAR))
    return out


# ── aggregation, present/ledgered, the processor ───────────────────────────────────────────────────

def _aggregate(ch: Any, start: dt.date, end: dt.date
               ) -> List[Tuple[str, dt.date, List[ExpiryNode]]]:
    """The ok iso_v1 expiry points for each (underlying, session_date) in [start, end], grouped and
    ordered by τ. Horizon rows are ignored; the windows are built from expiries only."""
    res = ch.query(
        f"SELECT underlying, session_date, expiry, tau_td, fit_total_var, support "
        f"FROM {IV_CURVE_TABLE} FINAL "
        "WHERE method = %(m)s AND kind = %(k)s AND status = %(s)s "
        "AND session_date BETWEEN %(a)s AND %(b)s "
        "ORDER BY underlying, session_date, tau_td "
        "SETTINGS do_not_merge_across_partitions_select_final = 1",
        parameters={"m": SOURCE_METHOD, "k": CURVE_KIND_EXPIRY, "s": CURVE_STATUS_OK,
                    "a": start, "b": end})
    groups: "dict[Tuple[str, dt.date], List[ExpiryNode]]" = defaultdict(list)
    order: List[Tuple[str, dt.date]] = []
    for r in res.result_rows:
        key = (str(r[0]), r[1])
        if key not in groups:
            order.append(key)
        groups[key].append(ExpiryNode(expiry=r[2], tau=int(r[3]), fit_total_var=float(r[4]),
                                      support=float(r[5])))
    return [(u, d, groups[(u, d)]) for (u, d) in order]


def _present_pairs(ch: Any, start: dt.date, end: dt.date) -> Set[Pair]:
    """Every (session_date, underlying) present in iv_curve_daily (method iso_v1) for [start, end] —
    the ledger must cover all of them, so a fully-processed range is a subset of the ledger."""
    res = ch.query(
        f"SELECT DISTINCT session_date, underlying FROM {IV_CURVE_TABLE} "
        "WHERE method = %(m)s AND session_date BETWEEN %(a)s AND %(b)s",
        parameters={"m": SOURCE_METHOD, "a": start, "b": end})
    return {(r[0], str(r[1]).upper()) for r in res.result_rows}


def _ledgered_pairs(ch: Any, start: dt.date, end: dt.date) -> Set[Pair]:
    """The (session_date, underlying) already marked done for this dataset. Read ONCE per backfill."""
    res = ch.query(
        f"SELECT DISTINCT session_date, ticker FROM {LEDGER_TABLE} FINAL "
        "WHERE dataset = %(d)s AND session_date BETWEEN %(a)s AND %(b)s",
        parameters={"d": LEDGER_DATASET, "a": start, "b": end})
    return {(r[0], str(r[1]).upper()) for r in res.result_rows}


def month_is_done(present: Set[Pair], ledgered: Set[Pair]) -> bool:
    """A month is fully processed when every (day, underlying) present in iv_curve_daily for it is
    already ledgered — pure, so the restart-skip rule is unit-tested without a client."""
    return bool(present) and present <= ledgered


def ledger_rows(present: Set[Pair], per_pair: Mapping[Pair, int]) -> List[List[Any]]:
    """One ledger row per present pair, rows_inserted = the iv_event rows written for it (0 if none)."""
    return [[day, ticker, int(per_pair.get((day, ticker), 0)), 0, LEDGER_DATASET, "", ""]
            for day, ticker in sorted(present)]


def _event_row(underlying: str, session_date: dt.date, w: Window,
               daytype_from: dt.date, daytype_to: dt.date) -> List[Any]:
    return [
        METHOD, underlying.upper(), session_date, w.start_expiry, w.end_expiry, w.start_td, w.end_td,
        w.eff_days, w.window_var, w.normal_daily_var, w.excess_var, w.excess_days, w.implied_move,
        w.support, w.own_earnings_date, w.own_earnings_source, w.status, SOURCE_METHOD,
        daytype_from, daytype_to,
    ]


def process_range(ch: Any, start: dt.date, end: dt.date,
                  present: Optional[Set[Pair]] = None) -> "Counter[str]":
    """Aggregate [start, end], build each (underlying, day)'s windows, insert the rows with ONE insert,
    then write the ledger (every present pair, rows_inserted = its row count). Ledger write is LAST, so
    a failure before it leaves no ledger row and a rerun redoes it (re-insert is safe under
    ReplacingMergeTree). Returns the status histogram over the written rows."""
    weights, daytype_from, daytype_to = _require_daytype(ch)
    groups = _aggregate(ch, start, end)
    today = dt.date.today()
    underlyings = {u.upper() for (u, _d, _e) in groups}
    earnings = _load_earnings(ch, underlyings, today)

    rows: List[List[Any]] = []
    counts: "Counter[str]" = Counter()
    per_pair: "Counter[Pair]" = Counter()
    for underlying, session_date, exps in groups:
        nodes = select_nodes(exps)
        day_weight = _day_weight_fn(weights, underlying, session_date)
        own = _own_earnings_fn(earnings.get(underlying.upper(), []))

        for w in build_day(session_date, nodes, day_weight=day_weight, list_days=trading_days_in,
                           own_earnings=own):
            counts[w.status] += 1
            rows.append(_event_row(underlying, session_date, w, daytype_from, daytype_to))
            per_pair[(session_date, underlying.upper())] += 1
    if rows:
        ch.insert(IV_EVENT_TABLE, rows, column_names=list(COLUMNS))

    if present is None:
        present = _present_pairs(ch, start, end)
    led = ledger_rows(present, per_pair)
    if led:
        ch.insert(LEDGER_TABLE, led, column_names=_LEDGER_COLUMNS)
    return counts


def _month_ranges(start: dt.date, end: dt.date) -> Iterator[Tuple[dt.date, dt.date]]:
    """Yield (month_start, month_end) clipped to [start, end], one calendar month at a time."""
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        nxt = dt.date(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
        yield max(cur, start), min(nxt - dt.timedelta(days=1), end)
        cur = nxt


def _summary(counts: Mapping[str, int]) -> str:
    """`status=n status=n` ordered by the status ladder, for the one-line-per-month/day summary."""
    return " ".join(f"{s}={counts.get(s, 0)}" for s in _STATUS_ORDER)


def backfill(ch: Any, get_client: Callable[[], Any], start: dt.date, end: dt.date) -> "Counter[str]":
    """Process [start, end] one calendar month per statement. ONE ledger read seeds the restart-skip:
    a month whose every present (day, underlying) is already ledgered is skipped. A lost ClickHouse
    connection rebuilds the client once (via `get_client`) and retries that month; a second failure
    propagates and stops the run, leaving no ledger for the month. Ctrl-C stops cleanly after the
    current month. DaytypeMissing propagates to the caller (no weights → nothing to write)."""
    reset_stop()
    ledgered = _ledgered_pairs(ch, start, end)
    box: List[Any] = [ch]
    total: "Counter[str]" = Counter()
    for ms, me in _month_ranges(start, end):
        if _stop_requested():
            print("[iv_event] stop requested — stopping cleanly after the current month")
            break
        t0 = time.perf_counter()
        result = _month_with_retry(box, get_client, ms, me, ledgered)
        secs = time.perf_counter() - t0
        if result is None:
            print(f"[iv_event] {ms} .. {me}: skipped (already ledgered)")
            continue
        total += result
        print(f"[iv_event] {ms} .. {me}: {_summary(result)}  ({secs:.1f}s)")
    return total


def _month_with_retry(box: List[Any], get_client: Callable[[], Any], ms: dt.date, me: dt.date,
                      ledgered: Set[Pair]) -> Optional["Counter[str]"]:
    """Process one month, or None if it is already fully ledgered. Rebuild the client once on a lost
    connection and retry; a second failure propagates."""
    try:
        return _month_once(box[0], ms, me, ledgered)
    except OperationalError as exc:
        log.info("iv_event %s..%s: lost ClickHouse connection (%s) — rebuilding client and retrying",
                 ms, me, exc)
        box[0] = get_client()
        return _month_once(box[0], ms, me, ledgered)


def _month_once(ch: Any, ms: dt.date, me: dt.date, ledgered: Set[Pair]) -> Optional["Counter[str]"]:
    present = _present_pairs(ch, ms, me)
    if month_is_done(present, ledgered):
        return None
    return process_range(ch, ms, me, present)


def _max_date(ch: Any, sql: str, parameters: Optional[Mapping[str, Any]] = None) -> Optional[dt.date]:
    res = ch.query(sql, parameters=dict(parameters) if parameters else None)
    val = res.result_rows[0][0] if res.result_rows else None
    # ClickHouse max() over no rows yields the Date epoch, not NULL; treat that as "nothing".
    if val is None or val <= dt.date(1970, 1, 1):
        return None
    assert isinstance(val, dt.date)
    return val


def daily(ch: Any) -> int:
    """Process every session_date from the newest iv_event_fwd_v1 ledger day (inclusive) to the newest
    iv_curve_daily day, through the same code path as backfill, limited to those days. Prints
    `wrote N rows`. Returns the rows written, or -1 (caller exits non-zero) when there is no ledger day
    yet or no day weights for the method — so the batch job reports the failure rather than starting a
    full-history run or writing unweighted windows."""
    newest_ledger = _max_date(
        ch, f"SELECT max(session_date) FROM {LEDGER_TABLE} FINAL WHERE dataset = %(d)s",
        {"d": LEDGER_DATASET})
    if newest_ledger is None:
        print("no iv_event ledger rows - run the backfill first")
        return -1
    try:
        _require_daytype(ch)
    except DaytypeMissing as exc:
        print(str(exc))
        return -1
    newest_curve = _max_date(
        ch, f"SELECT max(session_date) FROM {IV_CURVE_TABLE} WHERE method = %(m)s",
        {"m": SOURCE_METHOD})
    if newest_curve is None:
        print("no iv_event ledger rows - run the backfill first")
        return -1

    counts = process_range(ch, newest_ledger, newest_curve)
    n = sum(counts.values())
    print(f"wrote {n} rows")
    return n
