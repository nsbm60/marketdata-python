"""
The ClickHouse side of method atm_v1: the month/day aggregation over trading.option_trade, the
per-group solves, the single insert into trading.iv_term_daily, the ledger, and the backfill/daily
drivers.

One aggregation query per call collapses a date range down to the size-weighted moments of each
(underlying, session_date, expiry); NumPy (fit.py) solves and classifies each group; the rows go in
with one insert. The ledger (dataset iv_term_atm_v1) carries one row per (session_date, underlying)
PRESENT in option_trade for the range — including those that produced no iv_term row (rows_inserted
= 0) — so a restart can tell a fully-processed range from a partial one by a subset test.

Logging is one line per month/day at INFO; nothing per pair or per expiry.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections import Counter
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Set, Tuple

from clickhouse_connect.driver.exceptions import OperationalError

from . import (
    DELTA_HIGH, DELTA_LOW, GREEKS_METHODOLOGY, IV_TERM_TABLE, LEDGER_DATASET, LEDGER_TABLE, METHOD,
    OPTION_TRADE_TABLE, STATUS_FIT_FAILED, STATUS_FIT_OUT_OF_RANGE, STATUS_OK, STATUS_ONE_SIDED,
    STATUS_TOO_FEW_STRIKES, STATUS_TOO_FEW_TRADES, WINDOW_ET_HOUR,
)
from .fit import SmileAgg, SmileFit, classify

log = logging.getLogger("iv_term.term")

Pair = Tuple[dt.date, str]

# The iv_term_daily columns written from Python, in insert order. computed_at is the
# ReplacingMergeTree version and is left to the DDL DEFAULT (now64(3)), so a re-insert wins — exactly
# as the greeks writer leaves option_trade.ingested_at. The column-contract test keys on this.
COLUMNS: Tuple[str, ...] = (
    "method", "underlying", "session_date", "expiry", "status", "greeks_methodology",
    "atm_iv", "total_var", "slope", "curvature", "tte", "forward",
    "n_trades", "n_strikes", "contracts", "k_min", "k_max",
)

_LEDGER_COLUMNS = ["session_date", "ticker", "rows_inserted", "quote_failures", "dataset",
                   "gap_reason", "claimed_by"]

# Status display order (the ladder), for the one-line-per-month/day summary.
_STATUS_ORDER = (STATUS_OK, STATUS_TOO_FEW_TRADES, STATUS_TOO_FEW_STRIKES, STATUS_ONE_SIDED,
                 STATUS_FIT_FAILED, STATUS_FIT_OUT_OF_RANGE)

# Clean-stop flag: __main__ installs a SIGINT handler that sets it; backfill polls it at each month
# boundary and stops after the current month (a plain flag, set from a signal handler between
# bytecodes — no threading).
_STOP: List[bool] = [False]


def request_stop() -> None:
    _STOP[0] = True


def reset_stop() -> None:
    _STOP[0] = False


def _stop_requested() -> bool:
    return _STOP[0]


# ── the SQL symbol parse (OSI suffix is fixed width: YYMMDD + C|P + strike*1000 = 6+1+8 = 15) ──────

def symbol_parse_exprs(col: str) -> Dict[str, str]:
    """ClickHouse expressions that parse a bare OSI `col` into expiry / strike / right, matching
    greeks.occ.parse_occ. The root is 1–6 letters, so the 15-char suffix is anchored on the right:
    YYMMDD at length-14, C|P at length-8, the 8-digit strike*1000 at length-7. Returned as SQL
    fragments so they are defined once (and the parse test can run them read-only on literals)."""
    length = f"length({col})"
    expiry = (f"toDate(concat('20', substring({col}, {length} - 14, 2), '-', "
              f"substring({col}, {length} - 12, 2), '-', substring({col}, {length} - 10, 2)))")
    strike = f"toFloat64(substring({col}, {length} - 7, 8)) / 1000"
    right = f"substring({col}, {length} - 8, 1)"
    return {"expiry": expiry, "strike": strike, "right": right}


def _aggregate(ch: Any, start: dt.date, end: dt.date) -> List[Tuple[str, dt.date, dt.date, SmileAgg]]:
    """One aggregation over option_trade for session_date in [start, end]: the size-weighted moments
    and summary stats of each (underlying, session_date, expiry) group, over the qualifying rows.

    Qualifying row: v2 trade-basis greeks that succeeded; a standard OSI symbol (so the parse matches
    parse_occ's domain); the 15:00–16:00 ET window (ET hour == 15, i.e. 15:00 inclusive to 16:00
    exclusive); |delta| in the band; positive size; and a later expiry (same-day expiries excluded)."""
    e = symbol_parse_exprs("symbol")
    sql = f"""
WITH parsed AS (
    SELECT
        underlying,
        session_date,
        {e['expiry']} AS expiry,
        {e['strike']} AS strike,
        toFloat64(size) AS w,
        log({e['strike']} / greeks_forward) AS k,
        iv AS iv,
        greeks_tte AS tte,
        greeks_forward AS fwd
    FROM {OPTION_TRADE_TABLE} FINAL
    WHERE session_date BETWEEN %(a)s AND %(b)s
      AND greeks_status = 'success'
      AND greeks_methodology = %(m)s
      AND match(symbol, '^[A-Z]{{1,6}}[0-9]{{6}}[CP][0-9]{{8}}$')
      AND toHour(toTimezone(trade_ts, 'America/New_York')) = {WINDOW_ET_HOUR}
      AND abs(delta) BETWEEN %(dlo)s AND %(dhi)s
      AND size > 0
)
SELECT
    underlying, session_date, expiry,
    count()                      AS n_trades,
    uniqExact(strike)            AS n_strikes,
    sum(w)                       AS contracts,
    sum(w)                       AS sw,
    sum(w * k)                   AS swk,
    sum(w * k * k)               AS swk2,
    sum(w * k * k * k)           AS swk3,
    sum(w * k * k * k * k)       AS swk4,
    sum(w * iv)                  AS swv,
    sum(w * k * iv)              AS swkv,
    sum(w * k * k * iv)          AS swk2v,
    sum(w * tte)                 AS sw_tte,
    sum(w * fwd)                 AS sw_fwd,
    min(k)                       AS k_min,
    max(k)                       AS k_max,
    min(iv)                      AS iv_min,
    max(iv)                      AS iv_max
FROM parsed
WHERE expiry > session_date
GROUP BY underlying, session_date, expiry
SETTINGS do_not_merge_across_partitions_select_final = 1
"""
    res = ch.query(sql, parameters={"a": start, "b": end, "m": GREEKS_METHODOLOGY,
                                    "dlo": DELTA_LOW, "dhi": DELTA_HIGH})
    out: List[Tuple[str, dt.date, dt.date, SmileAgg]] = []
    for r in res.result_rows:
        underlying, session_date, expiry = str(r[0]), r[1], r[2]
        agg = SmileAgg(
            n_trades=int(r[3]), n_strikes=int(r[4]), contracts=float(r[5]), sw=float(r[6]),
            swk=float(r[7]), swk2=float(r[8]), swk3=float(r[9]), swk4=float(r[10]),
            swv=float(r[11]), swkv=float(r[12]), swk2v=float(r[13]),
            sw_tte=float(r[14]), sw_fwd=float(r[15]),
            k_min=float(r[16]), k_max=float(r[17]), iv_min=float(r[18]), iv_max=float(r[19]),
        )
        out.append((underlying, session_date, expiry, agg))
    return out


def _present_pairs(ch: Any, start: dt.date, end: dt.date) -> Set[Pair]:
    """Every (session_date, underlying) present in option_trade for [start, end] — the ledger must
    cover all of them (correction A), so a fully-processed range is a subset of the ledger."""
    res = ch.query(
        f"SELECT DISTINCT session_date, underlying FROM {OPTION_TRADE_TABLE} "
        "WHERE session_date BETWEEN %(a)s AND %(b)s",
        parameters={"a": start, "b": end})
    return {(r[0], str(r[1]).upper()) for r in res.result_rows}


def _ledgered_pairs(ch: Any, start: dt.date, end: dt.date) -> Set[Pair]:
    """The (session_date, underlying) already marked done for this dataset in [start, end]. Read ONCE
    at the start of a backfill, never per month."""
    res = ch.query(
        f"SELECT DISTINCT session_date, ticker FROM {LEDGER_TABLE} FINAL "
        "WHERE dataset = %(d)s AND session_date BETWEEN %(a)s AND %(b)s",
        parameters={"d": LEDGER_DATASET, "a": start, "b": end})
    return {(r[0], str(r[1]).upper()) for r in res.result_rows}


def month_is_done(present: Set[Pair], ledgered: Set[Pair]) -> bool:
    """A month is already fully processed when every (day, underlying) present in option_trade for it
    already has a ledger row — pure, so the restart-skip rule is unit-tested without a client."""
    return bool(present) and present <= ledgered


def ledger_rows(present: Set[Pair], per_pair: Mapping[Pair, int]) -> List[List[Any]]:
    """One ledger row per present pair, rows_inserted = the iv_term rows written for it (0 if none)."""
    return [[day, ticker, int(per_pair.get((day, ticker), 0)), 0, LEDGER_DATASET, "", ""]
            for day, ticker in sorted(present)]


def _build_row(underlying: str, session_date: dt.date, expiry: dt.date, agg: SmileAgg,
               fit: SmileFit) -> List[Any]:
    return [
        METHOD, underlying.upper(), session_date, expiry, fit.status, GREEKS_METHODOLOGY,
        fit.atm_iv, fit.total_var, fit.slope, fit.curvature, fit.tte, fit.forward,
        agg.n_trades, agg.n_strikes, agg.contracts, agg.k_min, agg.k_max,
    ]


def process_range(ch: Any, start: dt.date, end: dt.date,
                  present: Optional[Set[Pair]] = None) -> "Counter[str]":
    """Aggregate [start, end], solve and classify each group, insert the iv_term rows with ONE
    insert, then write the ledger (every present pair, rows_inserted = its iv_term row count). The
    ledger write is LAST, so a failure before it leaves no ledger row for the range and a rerun
    redoes it (re-insert is safe under ReplacingMergeTree). Returns the status histogram."""
    groups = _aggregate(ch, start, end)
    rows: List[List[Any]] = []
    counts: "Counter[str]" = Counter()
    per_pair: "Counter[Pair]" = Counter()
    for underlying, session_date, expiry, agg in groups:
        fit = classify(agg)
        counts[fit.status] += 1
        rows.append(_build_row(underlying, session_date, expiry, agg, fit))
        per_pair[(session_date, underlying.upper())] += 1
    if rows:
        ch.insert(IV_TERM_TABLE, rows, column_names=list(COLUMNS))

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
    """`status=n status=n …` ordered by the status ladder, for the one-line-per-month log."""
    return " ".join(f"{s}={counts.get(s, 0)}" for s in _STATUS_ORDER)


def backfill(ch: Any, get_client: Callable[[], Any], start: dt.date, end: dt.date) -> "Counter[str]":
    """Process [start, end] one calendar month per statement. ONE ledger read seeds the restart-skip:
    a month whose every present (day, underlying) is already ledgered is skipped. A lost ClickHouse
    connection rebuilds the client once (via `get_client`) and retries that month; a second failure
    propagates and stops the run, leaving no ledger for the month. Ctrl-C stops cleanly after the
    current month. `ch` is mutable per month on reconnect, so it is held in a one-element box."""
    reset_stop()
    ledgered = _ledgered_pairs(ch, start, end)
    box: List[Any] = [ch]
    total: "Counter[str]" = Counter()
    for ms, me in _month_ranges(start, end):
        if _stop_requested():
            print("[iv_term] stop requested — stopping cleanly after the current month")
            break
        t0 = time.perf_counter()
        result = _month_with_retry(box, get_client, ms, me, ledgered)
        secs = time.perf_counter() - t0
        if result is None:
            log.info("iv_term %s..%s: skipped (already ledgered)", ms, me)
            print(f"[iv_term] {ms} .. {me}: skipped (already ledgered)")
            continue
        total += result
        log.info("iv_term %s..%s: %s  (%.1fs)", ms, me, _summary(result), secs)
        print(f"[iv_term] {ms} .. {me}: {_summary(result)}  ({secs:.1f}s)")
    return total


def _month_with_retry(box: List[Any], get_client: Callable[[], Any], ms: dt.date, me: dt.date,
                      ledgered: Set[Pair]) -> Optional["Counter[str]"]:
    """Process one month, or return None if it is already fully ledgered. Rebuild the client once on
    a lost connection and retry; a second failure propagates."""
    try:
        return _month_once(box[0], ms, me, ledgered)
    except OperationalError as exc:
        log.info("iv_term %s..%s: lost ClickHouse connection (%s) — rebuilding client and retrying",
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
    """Process every session_date from the newest iv_term_atm_v1 ledger day (inclusive) to the newest
    day present in option_trade, through the same code path as backfill, limited to those days (not
    whole months). Prints `wrote N rows`. Returns the rows written, or -1 if there is no ledger day
    yet — the caller exits non-zero, so the batch job reports the failure rather than starting a
    full-history run (mirrors OptionArchiveNightly's empty-ledger handling)."""
    newest_ledger = _max_date(
        ch, f"SELECT max(session_date) FROM {LEDGER_TABLE} FINAL WHERE dataset = %(d)s",
        {"d": LEDGER_DATASET})
    if newest_ledger is None:
        print("no iv_term ledger rows - run the backfill first")
        return -1
    newest_trade = _max_date(ch, f"SELECT max(session_date) FROM {OPTION_TRADE_TABLE}")
    if newest_trade is None:
        print("no iv_term ledger rows - run the backfill first")
        return -1

    counts = process_range(ch, newest_ledger, newest_trade)
    n = sum(counts.values())
    log.info("iv_term daily %s..%s: %s", newest_ledger, newest_trade, _summary(counts))
    print(f"wrote {n} rows")
    return n
