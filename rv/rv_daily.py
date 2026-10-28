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
"""

from __future__ import annotations

import datetime as dt
import logging

log = logging.getLogger("rv.rv_daily")

# A few trading days of overlap so the first day of each month can see the prior complete session's
# close (lagInFrame over the partition); only rows on/after month start are emitted.
_LOOKBACK_DAYS = 10


def _with_core(lookback: dt.date, month_end: dt.date) -> str:
    """The shared WITH clause producing `withsplit`: one row per complete (symbol, trading_date) in
    [lookback, month_end] with the ordered grid prices, day open/close, prior close and split ratio."""
    lo, hi = lookback.isoformat(), month_end.isoformat()
    return f"""
WITH
grid AS (
    SELECT symbol, trading_date,
           toStartOfInterval(toTimezone(ts, 'America/New_York'), INTERVAL 5 minute) AS bucket,
           argMax(close, ts) AS px
    FROM trading.stock_bar
    WHERE period = '1m' AND session = 1
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
    WHERE period = '1m' AND session = 1
      AND trading_date >= toDate('{lo}') AND trading_date <= toDate('{hi}')
    GROUP BY symbol, trading_date
),
complete AS (
    SELECT symbol, trading_date, toUInt16(intDiv(any(expected_regular), 5)) AS expected_bars
    FROM trading.stock_bar_ledger FINAL
    WHERE status = 'complete'
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
          WHERE trading_date >= toDate('{lo}') AND trading_date <= toDate('{hi}'))
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
withsplit AS (
    SELECT w.*, s.ratio AS split_ratio
    FROM withprev w
    LEFT JOIN (SELECT symbol, execution_date, any(ratio) AS ratio
               FROM trading.split GROUP BY symbol, execution_date) s
      ON w.symbol = s.symbol AND w.trading_date = s.execution_date
)
"""


def _month_ranges(start: dt.date, end: dt.date):
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        nxt = dt.date(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
        yield max(cur, start), min(nxt - dt.timedelta(days=1), end)
        cur = nxt


def backfill(ch, start: dt.date, end: dt.date) -> int:
    """Compute rv_daily for every complete (symbol, day) in [start, end], one statement per month."""
    total = 0
    for ms, me in _month_ranges(start, end):
        lookback = ms - dt.timedelta(days=_LOOKBACK_DAYS)
        sql = f"""
INSERT INTO trading.rv_daily
    (symbol, trading_date, rv_intraday, rv_overnight, rv_total, n_bars, expected_bars, split_adjusted, source, computed_at)
{_with_core(lookback, me)}
SELECT
    symbol, trading_date,
    arraySum(arrayMap(x -> x * x, arrayDifference(arrayMap(p -> log(p), grid_px)))) AS rv_intraday,
    pow(log(day_open / (prev_close / coalesce(split_ratio, 1.0))), 2) AS rv_overnight,
    rv_intraday + rv_overnight AS rv_total,
    n_bars, expected_bars,
    toUInt8(split_ratio IS NOT NULL) AS split_adjusted,
    'har-rv' AS source,
    now64(3) AS computed_at
FROM withsplit
WHERE trading_date >= toDate('{ms.isoformat()}')
  AND prev_close > 0                              -- a previous complete session exists
  AND prev_complete_date = prev_trading_day       -- and it is the immediately previous trading day
SETTINGS join_use_nulls = 1
"""
        ch.command(sql)
        n = ch.query(
            f"SELECT count() FROM trading.rv_daily FINAL WHERE trading_date >= '{ms.isoformat()}' "
            f"AND trading_date <= '{me.isoformat()}'"
        ).result_rows[0][0]
        # Refuse non-finite or non-positive variance — fail fast, by name and date, never persist it.
        bad = ch.query(
            f"SELECT symbol, toString(trading_date), rv_total FROM trading.rv_daily FINAL "
            f"WHERE trading_date >= '{ms.isoformat()}' AND trading_date <= '{me.isoformat()}' "
            f"AND (NOT isFinite(rv_total) OR rv_total <= 0) ORDER BY symbol, trading_date LIMIT 100"
        ).result_rows
        if bad:
            lines = "\n".join(f"    {s:8s} {d}  rv_total={rv}" for s, d, rv in bad)
            raise RuntimeError(
                f"rv_daily produced non-finite/non-positive rv_total for {ms}..{me} "
                f"({len(bad)}+ rows shown):\n{lines}"
            )
        log.info("rv_daily %s..%s: %d rows", ms, me, n)
        print(f"[rv_daily] {ms} .. {me}: {n:,} rows")
        total += n
    return total


def stats(ch) -> dict:
    """Post-backfill summary: split-adjusted rows vs splits on these names, and rv_total spread."""
    row = ch.query(
        "SELECT count(), min(rv_total), median(rv_total), max(rv_total), sum(split_adjusted) "
        "FROM trading.rv_daily FINAL WHERE source = 'har-rv'"
    ).result_rows[0]
    splits = ch.query(
        "SELECT count() FROM (SELECT DISTINCT symbol, execution_date FROM trading.split "
        "WHERE execution_date >= '2016-01-01' AND symbol IN (SELECT DISTINCT symbol FROM trading.rv_daily))"
    ).result_rows[0][0]
    return {
        "rows": row[0], "rv_min": row[1], "rv_median": row[2], "rv_max": row[3],
        "split_adjusted": row[4], "splits_in_table": splits,
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
