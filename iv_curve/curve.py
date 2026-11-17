"""
The ClickHouse side of method iso_v1: the month/day aggregation over iv_term_daily, the per-pair
curve builds, the single insert into iv_curve_daily, the ledger, and the backfill/daily drivers.

Modelled on iv_term/term.py: one aggregation query per call collapses a date range to the ok ATM
expiries of each (underlying, session_date); iso.py builds the curve; the rows go in with one insert.
The ledger (dataset iv_curve_iso_v1) carries one row per (session_date, underlying) PRESENT in
iv_term_daily for the range — including pairs with no ok expiry (rows_inserted = 0) — so a restart
tells a fully-processed range from a partial one by a subset test.

Logging: the per-month/day summary is PRINTED once (not also logged); a reconnect is one log.info;
nothing per pair at INFO.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections import Counter, defaultdict
from typing import Any, Callable, Iterator, List, Mapping, Optional, Set, Tuple

from clickhouse_connect.driver.exceptions import OperationalError

from ml.shared.calendar import trading_days_between

from . import (
    IV_CURVE_TABLE, IV_TERM_TABLE, KIND_EXPIRY, KIND_HORIZON, LEDGER_DATASET, LEDGER_TABLE, METHOD,
    SOURCE_METHOD, STATUS_BEYOND_LAST_EXPIRY, STATUS_OK,
)
from .iso import CurvePoint, ExpiryInput, build_curve

log = logging.getLogger("iv_curve.curve")

Pair = Tuple[dt.date, str]

# The iv_curve_daily columns written from Python, in insert order. computed_at is the
# ReplacingMergeTree version, left to the DDL DEFAULT (now64(3)) so a re-insert wins — exactly as
# iv_term leaves computed_at. The column-contract test keys on this.
COLUMNS: Tuple[str, ...] = (
    "method", "underlying", "session_date", "kind", "tau_td", "expiry", "horizon_td", "status",
    "raw_total_var", "fit_total_var", "support", "bracket_lo_expiry", "bracket_hi_expiry",
    "source_method",
)

_LEDGER_COLUMNS = ["session_date", "ticker", "rows_inserted", "quote_failures", "dataset",
                   "gap_reason", "claimed_by"]

# Status display order, for the one-line summary.
_STATUS_ORDER = (STATUS_OK, STATUS_BEYOND_LAST_EXPIRY)

# Clean-stop flag: __main__ installs a SIGINT handler that sets it; backfill polls it at each month
# boundary and stops after the current month (a plain flag set from a signal handler — no threading).
_STOP: List[bool] = [False]


def request_stop() -> None:
    _STOP[0] = True


def reset_stop() -> None:
    _STOP[0] = False


def _stop_requested() -> bool:
    return _STOP[0]


def _aggregate(ch: Any, start: dt.date, end: dt.date
               ) -> List[Tuple[str, dt.date, List[Tuple[dt.date, float, float]]]]:
    """The ok ATM expiries for each (underlying, session_date) in [start, end], grouped. Each group
    is (underlying, session_date, [(expiry, total_var, n_trades), …]) ordered by expiry."""
    res = ch.query(
        f"SELECT underlying, session_date, expiry, total_var, n_trades "
        f"FROM {IV_TERM_TABLE} FINAL "
        "WHERE method = %(m)s AND status = %(s)s AND session_date BETWEEN %(a)s AND %(b)s "
        "ORDER BY underlying, session_date, expiry "
        "SETTINGS do_not_merge_across_partitions_select_final = 1",
        parameters={"m": SOURCE_METHOD, "s": STATUS_OK, "a": start, "b": end})
    groups: "dict[Tuple[str, dt.date], List[Tuple[dt.date, float, float]]]" = defaultdict(list)
    order: List[Tuple[str, dt.date]] = []
    for r in res.result_rows:
        key = (str(r[0]), r[1])
        if key not in groups:
            order.append(key)
        groups[key].append((r[2], float(r[3]), float(r[4])))
    return [(u, d, groups[(u, d)]) for (u, d) in order]


def _present_pairs(ch: Any, start: dt.date, end: dt.date) -> Set[Pair]:
    """Every (session_date, underlying) present in iv_term_daily (method atm_v1) for [start, end] —
    the ledger must cover all of them, so a fully-processed range is a subset of the ledger."""
    res = ch.query(
        f"SELECT DISTINCT session_date, underlying FROM {IV_TERM_TABLE} "
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
    """A month is fully processed when every (day, underlying) present in iv_term_daily for it is
    already ledgered — pure, so the restart-skip rule is unit-tested without a client."""
    return bool(present) and present <= ledgered


def ledger_rows(present: Set[Pair], per_pair: Mapping[Pair, int]) -> List[List[Any]]:
    """One ledger row per present pair, rows_inserted = the iv_curve rows written for it (0 if none)."""
    return [[day, ticker, int(per_pair.get((day, ticker), 0)), 0, LEDGER_DATASET, "", ""]
            for day, ticker in sorted(present)]


def _curve_row(underlying: str, session_date: dt.date, cp: CurvePoint) -> List[Any]:
    return [
        METHOD, underlying.upper(), session_date, cp.kind, cp.tau_td, cp.expiry, cp.horizon_td,
        cp.status, cp.raw_total_var, cp.fit_total_var, cp.support, cp.bracket_lo_expiry,
        cp.bracket_hi_expiry, SOURCE_METHOD,
    ]


def process_range(ch: Any, start: dt.date, end: dt.date,
                  present: Optional[Set[Pair]] = None) -> "Counter[str]":
    """Aggregate [start, end], build each (underlying, day)'s curve, insert the rows with ONE insert,
    then write the ledger (every present pair, rows_inserted = its row count). Ledger write is LAST,
    so a failure before it leaves no ledger row and a rerun redoes it (re-insert is safe under
    ReplacingMergeTree). Returns the status histogram over the written rows."""
    rows: List[List[Any]] = []
    counts: "Counter[str]" = Counter()
    per_pair: "Counter[Pair]" = Counter()
    for underlying, session_date, expiries in _aggregate(ch, start, end):
        inputs = [ExpiryInput(expiry=e, tau=trading_days_between(session_date, e),
                              raw_total_var=tv, n_trades=n)
                  for (e, tv, n) in expiries if e > session_date]
        for cp in build_curve(inputs):
            counts[cp.status] += 1
            rows.append(_curve_row(underlying, session_date, cp))
            per_pair[(session_date, underlying.upper())] += 1
    if rows:
        ch.insert(IV_CURVE_TABLE, rows, column_names=list(COLUMNS))

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
    current month."""
    reset_stop()
    ledgered = _ledgered_pairs(ch, start, end)
    box: List[Any] = [ch]
    total: "Counter[str]" = Counter()
    for ms, me in _month_ranges(start, end):
        if _stop_requested():
            print("[iv_curve] stop requested — stopping cleanly after the current month")
            break
        t0 = time.perf_counter()
        result = _month_with_retry(box, get_client, ms, me, ledgered)
        secs = time.perf_counter() - t0
        if result is None:
            print(f"[iv_curve] {ms} .. {me}: skipped (already ledgered)")
            continue
        total += result
        print(f"[iv_curve] {ms} .. {me}: {_summary(result)}  ({secs:.1f}s)")
    return total


def _month_with_retry(box: List[Any], get_client: Callable[[], Any], ms: dt.date, me: dt.date,
                      ledgered: Set[Pair]) -> Optional["Counter[str]"]:
    """Process one month, or None if it is already fully ledgered. Rebuild the client once on a lost
    connection and retry; a second failure propagates."""
    try:
        return _month_once(box[0], ms, me, ledgered)
    except OperationalError as exc:
        log.info("iv_curve %s..%s: lost ClickHouse connection (%s) — rebuilding client and retrying",
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
    """Process every session_date from the newest iv_curve_iso_v1 ledger day (inclusive) to the newest
    iv_term_daily day, through the same code path as backfill, limited to those days. Prints
    `wrote N rows`. Returns the rows written, or -1 if there is no ledger day yet — the caller exits
    non-zero so the batch job reports the failure rather than starting a full-history run."""
    newest_ledger = _max_date(
        ch, f"SELECT max(session_date) FROM {LEDGER_TABLE} FINAL WHERE dataset = %(d)s",
        {"d": LEDGER_DATASET})
    if newest_ledger is None:
        print("no iv_curve ledger rows - run the backfill first")
        return -1
    newest_term = _max_date(
        ch, f"SELECT max(session_date) FROM {IV_TERM_TABLE} WHERE method = %(m)s",
        {"m": SOURCE_METHOD})
    if newest_term is None:
        print("no iv_curve ledger rows - run the backfill first")
        return -1

    counts = process_range(ch, newest_ledger, newest_term)
    n = sum(counts.values())
    print(f"wrote {n} rows")
    return n
