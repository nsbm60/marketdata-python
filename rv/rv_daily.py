"""
Daily realized variance from the raw 1-minute bars (HAR edge spec §3), set-based SQL.

One statement per month (still set-based — no per-name or per-date loops; the month loop only keeps
a single statement small enough for drogon). Each statement, over all names at once:

  - resamples 1-minute regular-session closes onto the 5-minute grid (last close in each
    toStartOfInterval(ts, 5 minute) bucket);
  - folds the 09:30 open in as the first grid point, so a full day has 78 intraday returns;
  - rv_intraday = Σ squared 5-minute log returns; rv_overnight = squared log return from the prior
    complete session's last regular close to today's open; rv_total = their sum;
  - takes complete days and expected_bars from trading.stock_bar_ledger (the Scala calendar's
    persisted output), so gap / not_listed / failed days are excluded automatically;
  - divides the prior close by a trading.split ratio whose ex-date is today, before the overnight
    return, and sets split_adjusted.

A name's first complete session has no prior close and gets no row (no overnight return).

Each row also carries four intraday risk-decomposition measures (same 5-minute grid; r_1..r_n are
the intraday returns, n = n_bars; o is the signed overnight return, rv_overnight = o²):
  - rs_down / rs_up: downside / upside realized semivariance — Σ r_i² over negative / positive r_i,
    plus o² on the matching side of the overnight return. rs_down + rs_up = rv_total by construction.
  - bv_intraday: bipower variation (π/2)·(n/(n−1))·Σ_{i=2..n} |r_i|·|r_{i−1}|, intraday only, NULL
    when n < 2.
  - tq_intraday: tri-power quarticity n·μ⁻³·(n/(n−2))·Σ_{i=3..n} (|r_i|·|r_{i−1}|·|r_{i−2}|)^(4/3)
    with μ = 2^(2/3)·Γ(7/6)/Γ(1/2), intraday only, NULL when n < 3. Stored now so the jump test
    (RV vs BV, scaled by TQ/BV²) needs no second backfill.
These are Nullable(Float64) columns; the operator must add them with the ALTER TABLE named in
_require_new_columns before this job runs — the job never adds them itself. The pure-Python oracle
for all five intraday measures is rv/measures.py (checked by `python -m rv check`).
"""

from __future__ import annotations

import datetime as dt
import logging
import math

log = logging.getLogger("rv.rv_daily")

# A few trading days of overlap so the first day of each month can see the prior complete session's
# close (lagInFrame over the partition); only rows on/after month start are emitted.
_LOOKBACK_DAYS = 10

# μ = 2^(2/3)·Γ(7/6)/Γ(1/2) ≈ 0.8309 — the bipower/tri-power scaling constant for tq_intraday.
# Computed once here (and identically in rv/measures.py) so the constant is never transcribed.
_MU = 2 ** (2 / 3) * math.gamma(7 / 6) / math.gamma(1 / 2)

# The columns this extension adds; the operator must ALTER them in before the job runs.
_NEW_COLUMNS = ("rs_down", "rs_up", "bv_intraday", "tq_intraday")


def _symbol_filter(symbols) -> str:
    """` AND symbol IN ('A','B')` for a targeted rerun, or '' for all names."""
    if not symbols:
        return ""
    quoted = ", ".join("'" + s.replace("'", "") + "'" for s in symbols)
    return f" AND symbol IN ({quoted})"


def _with_core(lookback: dt.date, month_end: dt.date, sym: str = "") -> str:
    """The shared WITH clause producing `withsplit`: one row per complete (symbol, trading_date) in
    [lookback, month_end] with the ordered grid prices, day open/close, prior close and split ratio.
    `sym` is an optional ` AND symbol IN (...)` filter for a targeted rerun."""
    lo, hi = lookback.isoformat(), month_end.isoformat()
    return f"""
WITH
grid AS (
    SELECT symbol, trading_date,
           toStartOfInterval(toTimezone(ts, 'America/New_York'), INTERVAL 5 minute) AS bucket,
           argMax(close, ts) AS px
    FROM trading.stock_bar
    WHERE period = '1m' AND session = 1{sym}
      AND trading_date >= toDate('{lo}') AND trading_date <= toDate('{hi}')
    GROUP BY symbol, trading_date, bucket
),
daygrid AS (
    SELECT symbol, trading_date,
           arrayMap(t -> t.2, arraySort(t -> t.1, groupArray((toUnixTimestamp(bucket), px)))) AS closes,
           toUInt16(count()) AS n_bars
    FROM grid
    GROUP BY symbol, trading_date
),
oc AS (
    SELECT symbol, trading_date, argMin(open, ts) AS day_open, argMax(close, ts) AS day_close
    FROM trading.stock_bar
    WHERE period = '1m' AND session = 1{sym}
      AND trading_date >= toDate('{lo}') AND trading_date <= toDate('{hi}')
    GROUP BY symbol, trading_date
),
complete AS (
    SELECT symbol, trading_date, toUInt16(intDiv(any(expected_regular), 5)) AS expected_bars
    FROM trading.stock_bar_ledger FINAL
    WHERE status = 'complete'{sym}
      AND trading_date >= toDate('{lo}') AND trading_date <= toDate('{hi}')
    GROUP BY symbol, trading_date
),
merged AS (
    SELECT d.symbol AS symbol, d.trading_date AS trading_date, d.n_bars AS n_bars,
           c.expected_bars AS expected_bars, oc.day_open AS day_open, oc.day_close AS day_close,
           arrayConcat([oc.day_open], d.closes) AS grid_px
    FROM daygrid d
    INNER JOIN oc ON d.symbol = oc.symbol AND d.trading_date = oc.trading_date
    INNER JOIN complete c ON d.symbol = c.symbol AND d.trading_date = c.trading_date
),
ledger_days AS (
    -- each name's previous trading day, from its ledger rows whatever their status (complete,
    -- failed, gap, not_listed) — so a failed/gap day still counts as the previous session.
    SELECT symbol, trading_date,
           lagInFrame(trading_date) OVER (PARTITION BY symbol ORDER BY trading_date) AS prev_trading_day
    FROM (SELECT DISTINCT symbol, trading_date FROM trading.stock_bar_ledger FINAL
          WHERE trading_date >= toDate('{lo}') AND trading_date <= toDate('{hi}'){sym})
),
withprev AS (
    SELECT m.symbol AS symbol, m.trading_date AS trading_date, m.n_bars AS n_bars,
           m.expected_bars AS expected_bars, m.day_open AS day_open, m.day_close AS day_close,
           m.grid_px AS grid_px,
           lagInFrame(m.day_close) OVER (PARTITION BY m.symbol ORDER BY m.trading_date) AS prev_close,
           lagInFrame(m.trading_date) OVER (PARTITION BY m.symbol ORDER BY m.trading_date) AS prev_complete_date,
           ld.prev_trading_day AS prev_trading_day
    FROM merged m
    LEFT JOIN ledger_days ld ON m.symbol = ld.symbol AND m.trading_date = ld.trading_date
),
divs AS (
    -- cash dividends to subtract from the prior close on the ex-date (USD only, regular + special).
    -- Named `divs`, not `div` — `div` is a ClickHouse built-in (integer division).
    SELECT symbol, ex_date, sum(amount) AS div_usd
    FROM trading.dividend
    WHERE currency = 'USD'
      AND ex_date >= toDate('{lo}') AND ex_date <= toDate('{hi}'){sym}
    GROUP BY symbol, ex_date
),
withsplit AS (
    SELECT w.symbol AS symbol, w.trading_date AS trading_date, w.n_bars AS n_bars,
           w.expected_bars AS expected_bars, w.day_open AS day_open, w.day_close AS day_close,
           w.grid_px AS grid_px, w.prev_close AS prev_close,
           w.prev_complete_date AS prev_complete_date, w.prev_trading_day AS prev_trading_day,
           s.ratio AS split_ratio, d.div_usd AS div_usd, hr.rvs AS has_rv
    FROM withprev w
    LEFT JOIN (SELECT symbol, execution_date, any(ratio) AS ratio
               FROM trading.split_effective GROUP BY symbol, execution_date) s
      ON w.symbol = s.symbol AND w.trading_date = s.execution_date
    LEFT JOIN divs d ON w.symbol = d.symbol AND w.trading_date = d.ex_date
    -- Per-(symbol, trading_date) existence of an rv_daily row, so `incremental` can emit every
    -- MISSING complete day (has_rv IS NULL), not just days after a name's latest rv date — history
    -- reloaded underneath an existing name is filled too (TEVA). backfill ignores this (recomputes).
    LEFT JOIN (SELECT DISTINCT symbol AS rvs, trading_date AS rvd FROM trading.rv_daily FINAL
               WHERE source = 'har-rv'{sym}) hr
      ON w.symbol = hr.rvs AND w.trading_date = hr.rvd
)
"""


def _month_ranges(start: dt.date, end: dt.date):
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        nxt = dt.date(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
        yield max(cur, start), min(nxt - dt.timedelta(days=1), end)
        cur = nxt


def _insert_month(ch, ms: dt.date, me: dt.date, sym: str, extra_where: str = "") -> int:
    """Run one month's set-based INSERT over [ms, me] (with a `_LOOKBACK_DAYS` warm-up so the first
    day sees its prior close), then count and sanity-check the rows in range. `extra_where` is
    appended to the emit filter — `incremental` uses it to emit only days after each name's latest
    rv_daily row. Returns the row count in [ms, me]."""
    lookback = ms - dt.timedelta(days=_LOOKBACK_DAYS)
    sql = f"""
INSERT INTO trading.rv_daily
    (symbol, trading_date, rv_intraday, rv_overnight, rv_total, rs_down, rs_up, bv_intraday, tq_intraday,
     n_bars, expected_bars, split_adjusted, source, computed_at)
{_with_core(lookback, me, sym)},
arrs AS (
    -- The intraday 5-minute log returns r_1..r_n (leading log-grid point dropped, NOT the leading 0
    -- of arrayDifference) and the signed overnight return o (rv_overnight = o²), computed once here.
    SELECT withsplit.*,
           arrayPopFront(arrayDifference(arrayMap(p -> log(p), grid_px))) AS rets,
           log(day_open / ((prev_close / coalesce(split_ratio, 1.0)) - coalesce(div_usd, 0.0))) AS overnight_ret
    FROM withsplit
)
SELECT
    symbol, trading_date,
    arraySum(arrayMap(x -> x * x, rets)) AS rv_intraday,          -- Σ r_i² (identical to arrayDifference form)
    pow(overnight_ret, 2) AS rv_overnight,
    rv_intraday + rv_overnight AS rv_total,
    arraySum(arrayMap(x -> if(x < 0, x * x, 0.0), rets)) + if(overnight_ret < 0, rv_overnight, 0.0) AS rs_down,
    arraySum(arrayMap(x -> if(x > 0, x * x, 0.0), rets)) + if(overnight_ret > 0, rv_overnight, 0.0) AS rs_up,
    if(n_bars < 2, NULL,
       (pi() / 2) * (toFloat64(n_bars) / (toFloat64(n_bars) - 1))
       * arraySum(arrayMap((a, b) -> abs(a) * abs(b), arrayPopFront(rets), arrayPopBack(rets)))) AS bv_intraday,
    if(n_bars < 3, NULL,
       toFloat64(n_bars) * pow({_MU!r}, -3) * (toFloat64(n_bars) / (toFloat64(n_bars) - 2))
       * arraySum(arrayMap((a, b, c) -> pow(abs(a), 4.0 / 3) * pow(abs(b), 4.0 / 3) * pow(abs(c), 4.0 / 3),
                           arrayPopFront(arrayPopFront(rets)),
                           arrayPopFront(arrayPopBack(rets)),
                           arrayPopBack(arrayPopBack(rets))))) AS tq_intraday,
    n_bars, expected_bars,
    toUInt8(split_ratio IS NOT NULL) AS split_adjusted,
    'har-rv' AS source,
    now64(3) AS computed_at
FROM arrs
WHERE trading_date >= toDate('{ms.isoformat()}')
  AND prev_close > 0                              -- a previous complete session exists
  AND prev_complete_date = prev_trading_day       -- and it is the immediately previous trading day
  {extra_where}
SETTINGS join_use_nulls = 1
"""
    ch.command(sql)
    n = ch.query(
        f"SELECT count() FROM trading.rv_daily FINAL WHERE trading_date >= '{ms.isoformat()}' "
        f"AND trading_date <= '{me.isoformat()}'{sym}"
    ).result_rows[0][0]
    # Refuse non-finite or negative variance — fail fast, by name and date, never persist it.
    # Zero is a real value (a thinly-traded day where every price equalled the prior close); it
    # is stored and excluded from the fit by load_series's rv_total > 0.
    bad = ch.query(
        f"SELECT symbol, toString(trading_date), rv_total FROM trading.rv_daily FINAL "
        f"WHERE trading_date >= '{ms.isoformat()}' AND trading_date <= '{me.isoformat()}'{sym} "
        f"AND (NOT isFinite(rv_total) OR rv_total < 0) ORDER BY symbol, trading_date LIMIT 100"
    ).result_rows
    if bad:
        lines = "\n".join(f"    {s:8s} {d}  rv_total={rv}" for s, d, rv in bad)
        raise RuntimeError(
            f"rv_daily produced non-finite/negative rv_total for {ms}..{me} "
            f"({len(bad)}+ rows shown):\n{lines}"
        )
    # Same fail-fast, by name and date, for the semivariance / bipower / quarticity columns: refuse a
    # non-finite or negative value, a semivariance split that doesn't sum to rv_total, or a NULL
    # bv/tq where n_bars >= 3 (both must be present there). Print up to 100 offenders and raise.
    bad_new = ch.query(
        f"SELECT symbol, toString(trading_date), rs_down, rs_up, bv_intraday, tq_intraday, rv_total, n_bars "
        f"FROM trading.rv_daily FINAL "
        f"WHERE trading_date >= '{ms.isoformat()}' AND trading_date <= '{me.isoformat()}'{sym} AND ("
        f"  NOT isFinite(rs_down) OR rs_down < 0"
        f"  OR NOT isFinite(rs_up) OR rs_up < 0"
        f"  OR (bv_intraday IS NOT NULL AND (NOT isFinite(bv_intraday) OR bv_intraday < 0))"
        f"  OR (tq_intraday IS NOT NULL AND (NOT isFinite(tq_intraday) OR tq_intraday < 0))"
        f"  OR abs(rs_down + rs_up - rv_total) > 1e-12 + 1e-9 * rv_total"
        f"  OR ((bv_intraday IS NULL OR tq_intraday IS NULL) AND n_bars >= 3)"
        f") ORDER BY symbol, trading_date LIMIT 100"
    ).result_rows
    if bad_new:
        lines = "\n".join(
            f"    {s:8s} {d}  rs_down={rd} rs_up={ru} bv={bv} tq={tq} rv_total={rv} n_bars={nb}"
            for s, d, rd, ru, bv, tq, rv, nb in bad_new
        )
        raise RuntimeError(
            f"rv_daily produced invalid semivariance/bipower/quarticity for {ms}..{me} "
            f"({len(bad_new)}+ rows shown):\n{lines}"
        )
    log.info("rv_daily %s..%s: %d rows", ms, me, n)
    print(f"[rv_daily] {ms} .. {me}: {n:,} rows")
    return n


def _require_new_columns(ch) -> None:
    """Fail fast if trading.rv_daily is missing the semivariance / bipower / quarticity columns. The
    live table pre-dates them, so the operator adds them once with the ALTER TABLE below; CREATE
    TABLE IF NOT EXISTS (schema.py) can't, and this job never adds them itself."""
    quoted = ", ".join("'" + c + "'" for c in _NEW_COLUMNS)
    present = {r[0] for r in ch.query(
        f"SELECT name FROM system.columns WHERE database = 'trading' AND table = 'rv_daily' "
        f"AND name IN ({quoted})"
    ).result_rows}
    missing = [c for c in _NEW_COLUMNS if c not in present]
    if missing:
        raise RuntimeError(
            "trading.rv_daily is missing column(s): " + ", ".join(missing) + ".\n"
            "Run this ALTER TABLE first (the operator runs all DDL — this job does not add them):\n"
            "    ALTER TABLE trading.rv_daily\n"
            "        ADD COLUMN IF NOT EXISTS rs_down     Nullable(Float64) AFTER rv_total,\n"
            "        ADD COLUMN IF NOT EXISTS rs_up       Nullable(Float64) AFTER rs_down,\n"
            "        ADD COLUMN IF NOT EXISTS bv_intraday Nullable(Float64) AFTER rs_up,\n"
            "        ADD COLUMN IF NOT EXISTS tq_intraday Nullable(Float64) AFTER bv_intraday"
        )


def backfill(ch, start: dt.date, end: dt.date, symbols=None) -> int:
    """Compute rv_daily for every complete (symbol, day) in [start, end], one statement per month.
    `symbols` restricts to a targeted rerun (e.g. after a cut); None means all names."""
    _require_new_columns(ch)
    sym = _symbol_filter(symbols)
    total = 0
    for ms, me in _month_ranges(start, end):
        total += _insert_month(ch, ms, me, sym)
    return total


def incremental(ch, symbols=None) -> int:
    """The daily catch-up (HAR edge spec §4), driven by the data, never the clock: compute rv_daily
    for every complete ledger day that has NO rv_daily row yet (`has_rv IS NULL` in `_with_core`), up
    to the latest complete ledger day — not just days after a name's latest rv date. So history
    reloaded underneath an existing name (TEVA: 2,708 complete ledger days, 33 rv rows) is filled,
    and a brand-new name is computed, not skipped. The rules are unchanged: a row is written only
    when the previous complete day is the previous trading day (`_insert_month`), so days marked
    `failed` (including ExcludeRvDay exclusions) stay excluded and never let the overnight span a gap.

    The month scan starts at the earliest *computable* missing day — the earliest missing complete
    day whose previous complete day is its previous trading day. Once a full backfill has filled the
    history, only recent days (or none) are missing, so the daily run is cheap again; the permanent
    non-rows (each name's first complete day, and days after a `failed` block) are not treated as a
    gap because they fail the overnight rule."""
    _require_new_columns(ch)
    sym = _symbol_filter(symbols)

    end = ch.query(
        f"SELECT max(trading_date) FROM trading.stock_bar_ledger FINAL WHERE status = 'complete'{sym}"
    ).result_rows[0][0]
    if end is None:
        print("[rv_daily] no complete ledger days — nothing to compute")
        return 0

    # Earliest missing day that is actually computable: a complete day whose previous complete day is
    # its previous trading day (overnight-eligible) and that has no rv_daily row yet. The epoch guard
    # drops each name's first-ever complete day — lagInFrame returns 1970-01-01 (not NULL) for a
    # partition's first row, so without it prev_complete == prev_any == epoch would mark that
    # never-computable day as the start and drag the scan back to 2016 on every normal daily run.
    start = ch.query(
        f"""WITH
  allday AS (SELECT symbol, trading_date,
               lagInFrame(trading_date) OVER (PARTITION BY symbol ORDER BY trading_date) AS prev_any
             FROM (SELECT DISTINCT symbol, trading_date FROM trading.stock_bar_ledger FINAL
                   WHERE 1{sym})),
  comp AS (SELECT symbol, trading_date,
             lagInFrame(trading_date) OVER (PARTITION BY symbol ORDER BY trading_date) AS prev_complete
           FROM (SELECT symbol, trading_date FROM trading.stock_bar_ledger FINAL
                 WHERE status = 'complete'{sym}))
SELECT minOrNull(c.trading_date) FROM comp c
INNER JOIN allday a ON c.symbol = a.symbol AND c.trading_date = a.trading_date
WHERE c.prev_complete != toDate('1970-01-01')   -- not a name's first-ever day (lagInFrame -> epoch)
  AND c.prev_complete = a.prev_any
  AND (c.symbol, c.trading_date) NOT IN
      (SELECT symbol, trading_date FROM trading.rv_daily FINAL WHERE source = 'har-rv'{sym})"""
    ).result_rows[0][0]
    # minOrNull returns NULL (-> None) when no row matches, so an empty result never becomes a start.
    # (Plain min() returns the Date epoch 1970-01-01 for an empty match, which walked the scan from
    # 1970 through every empty month.) No computable missing day -> skip the rv step; forecast.daily
    # goes straight on to the refit/forecast.
    if start is None:
        print(f"[rv_daily] no missing computable days — current through {end}; skipping to refit/forecast")
        return 0
    # Belt and braces: an epoch (or older) date is never a real trading day. If one ever reaches here
    # it means the missing-days query is broken — fail loudly rather than walk the scan from 1970.
    if start <= dt.date(1970, 1, 1):
        raise RuntimeError(
            f"[rv_daily] start query produced a non-real date ({start}); refusing to start the scan "
            f"from the epoch — the missing-days query is broken"
        )

    print(f"[rv_daily] incremental {start} .. {end} (every complete ledger day with no rv_daily row)")
    extra = "AND has_rv IS NULL"
    total = 0
    for ms, me in _month_ranges(start, end):
        total += _insert_month(ch, ms, me, sym, extra)
    return total


def zero_rv_by_name(ch) -> list[tuple]:
    """(symbol, count) for every name with any zero-RV day — a real flat session, kept but excluded
    from the fit. Returned most-frequent first."""
    return ch.query(
        "SELECT symbol, count() AS n FROM trading.rv_daily FINAL "
        "WHERE source = 'har-rv' AND rv_total = 0 GROUP BY symbol ORDER BY n DESC, symbol"
    ).result_rows


def stats(ch) -> dict:
    """Post-backfill summary: split-adjusted rows vs splits on these names, and rv_total spread."""
    row = ch.query(
        "SELECT count(), min(rv_total), median(rv_total), max(rv_total), sum(split_adjusted), "
        "median(if(rv_intraday > 0 AND bv_intraday IS NOT NULL, bv_intraday / rv_intraday, NULL)), "
        "avg(rs_down > rs_up) "
        "FROM trading.rv_daily FINAL WHERE source = 'har-rv'"
    ).result_rows[0]
    splits = ch.query(
        "SELECT count() FROM (SELECT DISTINCT symbol, execution_date FROM trading.split_effective "
        "WHERE execution_date >= '2016-01-01' AND symbol IN (SELECT DISTINCT symbol FROM trading.rv_daily))"
    ).result_rows[0][0]
    return {
        "rows": row[0], "rv_min": row[1], "rv_median": row[2], "rv_max": row[3],
        "split_adjusted": row[4], "splits_in_table": splits,
        "median_bv_rv": row[5], "share_rs_down_gt_up": row[6],
    }


def gap_dropped_count(ch) -> int:
    """
    How many complete days the "overnight must not span a gap" rule drops: complete days that have
    a previous complete day (the old rule would emit) but whose previous complete day is not their
    previous trading day (a failed/gap/not_listed session sits between). Ledger-only, one pass.
    """
    sql = """
WITH
all_days AS (
    SELECT symbol, trading_date,
           lagInFrame(trading_date) OVER (PARTITION BY symbol ORDER BY trading_date) AS prev_any
    FROM (SELECT DISTINCT symbol, trading_date FROM trading.stock_bar_ledger FINAL)
),
complete_days AS (
    SELECT symbol, trading_date,
           lagInFrame(trading_date) OVER (PARTITION BY symbol ORDER BY trading_date) AS prev_complete
    FROM (SELECT DISTINCT symbol, trading_date FROM trading.stock_bar_ledger FINAL WHERE status = 'complete')
)
SELECT count()
FROM complete_days c
INNER JOIN all_days a ON c.symbol = a.symbol AND c.trading_date = a.trading_date
WHERE c.prev_complete != toDate('1970-01-01') AND c.prev_complete != a.prev_any
"""
    return ch.query(sql).result_rows[0][0]


def split_proof(ch) -> list[tuple]:
    """rv_overnight / rv_total on AAPL 2020-08-31 (4:1) and NVDA 2024-06-10 (10:1). If the split is
    handled, the overnight move (sqrt of rv_overnight) is ordinary-sized, not a split-sized jump."""
    sql = """
SELECT symbol, toString(trading_date) AS d, rv_overnight, sqrt(rv_overnight) AS overnight_abs_ret,
       rv_intraday, rv_total, split_adjusted
FROM trading.rv_daily FINAL
WHERE (symbol = 'AAPL' AND trading_date = '2020-08-31')
   OR (symbol = 'NVDA' AND trading_date = '2024-06-10')
ORDER BY symbol
"""
    return ch.query(sql).result_rows


# ── read-only verification: pure-Python oracle vs stored rv_daily rows (writes nothing) ──

_CHECK_CASES = [
    ("AAPL", dt.date(2024, 8, 5)),
    ("SPY", dt.date(2024, 8, 5)),
    ("SPY", dt.date(2025, 4, 9)),
    ("LYFT", dt.date(2024, 2, 14)),
    ("NVDA", dt.date(2024, 6, 10)),   # 10:1 split day
]
_CHECK_COLS = ("rv_intraday", "rs_down", "rs_up", "bv_intraday", "tq_intraday")


def _agree(ref, stored) -> tuple[bool, float | None]:
    """(agrees, relative diff). Agreement = within 1e-6 relative or 1e-15 absolute. rel is None for a
    matching NULL on both sides; a one-sided NULL is a mismatch with rel = inf. The tolerance is 1e-6
    (not tighter) because ClickHouse's log and Python's math.log differ by ~1e-9 absolute per call
    (e.g. log(201.37): CH 5.3051440126620895 vs Python 5.305144011890314), so each 5-minute return
    carries ~1e-9 absolute error that the sums and the 4/3 powers in tq_intraday amplify (measured
    worst: DFDV 2024-08-05 tq_intraday, 1.99e-07 relative)."""
    if ref is None and stored is None:
        return True, None
    if ref is None or stored is None:
        return False, float("inf")
    diff = abs(float(ref) - float(stored))
    rel = diff / abs(float(stored)) if stored != 0 else diff
    return (diff <= 1e-15 or rel <= 1e-6), rel


def _fmt(x) -> str:
    return "NULL" if x is None else f"{float(x):.6e}"


def check(ch) -> bool:
    """Read-only (writes nothing). For a fixed set of (symbol, date) cases: rebuild the 5-minute grid
    from the raw 1m bars the same way the insert does, recompute the five intraday measures with the
    pure-Python oracle (rv/measures.py), and compare column by column to the stored rv_daily row.
    Prints each case, the relative difference per column, and bv/rv_intraday and rs_down/rv_total so
    the jump and sign behaviour is visible. Ends with one PASS/FAIL line. Returns True iff every
    column on every case agrees to 1e-6 relative (or 1e-15 absolute).

    The tolerance is 1e-6, not tighter, because ClickHouse's log and Python's math.log differ by
    ~1e-9 absolute per call (measured — e.g. log(201.37): CH 5.3051440126620895 vs Python
    5.305144011890314), so each 5-minute return carries ~1e-9 absolute error that the squared sums
    and the 4/3 powers in tq_intraday amplify (measured worst: DFDV 2024-08-05 tq_intraday,
    1.99e-07 relative). This is a log-implementation difference, not a formula difference."""
    from . import measures

    cases = list(_CHECK_CASES)
    # The three complete days with the fewest n_bars (>= 3) on 2024-08-05 — exercise the small-n tail.
    for s, d in ch.query(
        "SELECT symbol, trading_date FROM trading.rv_daily FINAL "
        "WHERE source = 'har-rv' AND trading_date = '2024-08-05' AND n_bars >= 3 "
        "ORDER BY n_bars ASC, symbol LIMIT 3"
    ).result_rows:
        cases.append((s, d))

    worst = (0.0, None, None)  # (rel diff, "symbol date", column)
    all_ok = True

    for sym_name, date in cases:
        symf = f" AND symbol IN ('{sym_name}')"
        lo = date - dt.timedelta(days=_LOOKBACK_DAYS)
        grid_rows = ch.query(
            f"{_with_core(lo, date, symf)}"
            f"SELECT grid_px, "
            f"log(day_open / ((prev_close / coalesce(split_ratio, 1.0)) - coalesce(div_usd, 0.0))) AS overnight_ret, "
            f"n_bars "
            f"FROM withsplit "
            f"WHERE trading_date = toDate('{date.isoformat()}') "
            f"  AND prev_close > 0 AND prev_complete_date = prev_trading_day "
            f"SETTINGS join_use_nulls = 1"
        ).result_rows
        stored_rows = ch.query(
            f"SELECT rv_intraday, rs_down, rs_up, bv_intraday, tq_intraday, rv_total, n_bars "
            f"FROM trading.rv_daily FINAL WHERE source = 'har-rv' AND symbol = '{sym_name}' "
            f"AND trading_date = toDate('{date.isoformat()}')"
        ).result_rows
        if not grid_rows or not stored_rows:
            missing = ", ".join(m for m, cond in (("no grid", grid_rows), ("no stored row", stored_rows)) if not cond)
            print(f"[check] {sym_name} {date}: MISSING ({missing}) — cannot compare")
            all_ok = False
            continue

        grid_px, overnight_ret, n_bars = grid_rows[0]
        ref = measures.day_measures([float(p) for p in grid_px], float(overnight_ret))
        s_rv, s_rs_down, s_rs_up, s_bv, s_tq, s_rv_total, _ = stored_rows[0]
        refd = {"rv_intraday": ref.rv_intraday, "rs_down": ref.rs_down, "rs_up": ref.rs_up,
                "bv_intraday": ref.bv_intraday, "tq_intraday": ref.tq_intraday}
        stored = {"rv_intraday": s_rv, "rs_down": s_rs_down, "rs_up": s_rs_up,
                  "bv_intraday": s_bv, "tq_intraday": s_tq}

        print(f"[check] {sym_name} {date}  n_bars={n_bars}")
        print(f"    {'column':12s} {'reference':>16s} {'stored':>16s} {'rel diff':>12s}")
        for c in _CHECK_COLS:
            ok, rel = _agree(refd[c], stored[c])
            all_ok = all_ok and ok
            rel_str = "null-match" if rel is None else f"{rel:.2e}"
            flag = "" if ok else "  <-- MISMATCH"
            print(f"    {c:12s} {_fmt(refd[c]):>16s} {_fmt(stored[c]):>16s} {rel_str:>12s}{flag}")
            if rel is not None and rel > worst[0]:
                worst = (rel, f"{sym_name} {date}", c)
        bv_ratio = (s_bv / s_rv) if (s_bv is not None and s_rv) else None
        down_ratio = (s_rs_down / s_rv_total) if s_rv_total else None
        print(f"    bv/rv_intraday = {_fmt(bv_ratio)}   rs_down/rv_total = {_fmt(down_ratio)}")

    if all_ok:
        print("PASS — every column agrees to 1e-6 relative (or 1e-15 absolute)")
    else:
        where = f"{worst[1]} {worst[2]} (rel {worst[0]:.2e})" if worst[1] else "missing row(s)"
        print(f"FAIL — worst: {where}")
    return all_ok
