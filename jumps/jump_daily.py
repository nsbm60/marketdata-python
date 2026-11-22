"""
The ClickHouse side of method bns_v1: the set-based per-month INSERT into trading.jump_daily, the
fail-fast checks, the backfill summaries and the read-only `show`. One statement per month like
rv_daily (the month loop only keeps each statement small); each statement, over all names at once:

  - recomputes the signed overnight return through rv_daily's own `_with_core` + `OVERNIGHT_RET_SQL`
    (NOT a copy — the same return rv_daily squares), over [month_start − _JUMP_LOOKBACK_DAYS,
    month_end] so the trailing overnight medians are complete;
  - INNER JOINs trading.rv_daily (source har-rv) — this fixes the row set (one row per rv_daily row,
    nothing more) and supplies rv_intraday / bv_intraday / tq_intraday / n_bars;
  - LEFT JOINs SPY's overnight return that date for the market-adjusted move;
  - the two trailing medians are window functions (quantileExactInclusive(0.5) over the 66 preceding
    sessions, excluding today);
  - inserts rows with trading_date on/after month_start.

SPY is always recomputed (even on a targeted --symbols rerun) so the adjustment always has a market
proxy; the rv_daily join is restricted to the requested names, so a targeted rerun emits only those.
Re-insert-safe under ReplacingMergeTree(computed_at). Quiet logging: one line per month, nothing per
name.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, List, Optional

from ml.shared.calendar import previous_trading_day

from rv.rv_daily import OVERNIGHT_RET_SQL, _with_core

from . import (
    EARNINGS_HISTORY_TABLE, JUMP_DAILY_TABLE, METHOD, ON_MIN_HISTORY, ON_WINDOW, RV_DAILY_TABLE,
    RV_SOURCE, SPY_SYMBOL, THETA, Z_INTRADAY, Z_OVERNIGHT, _JUMP_LOOKBACK_DAYS,
)

log = logging.getLogger("jumps.jump_daily")

# The jump_daily columns written from Python, in insert order = DDL order. Unlike iv_event, the DDL
# has no DEFAULT on computed_at, so the INSERT…SELECT supplies it (now64(3)) and it is in COLUMNS.
# The column-contract test (tests/jumps/test_jump_columns.py) keys on this being the full DDL set.
COLUMNS = (
    "method", "symbol", "trading_date", "n_bars", "rv_intraday", "bv_intraday", "jump_z",
    "intraday_jump", "jump_var", "cont_var", "overnight_ret", "overnight_z", "overnight_jump",
    "overnight_ret_adj", "overnight_z_adj", "overnight_jump_adj", "computed_at",
)


def _symbol_in(symbols) -> str:
    """` AND symbol IN ('A','B')` for a targeted rerun, or '' for all names."""
    if not symbols:
        return ""
    quoted = ", ".join("'" + s.replace("'", "") + "'" for s in symbols)
    return f" AND symbol IN ({quoted})"


def _require_table(ch: Any) -> None:
    """Fail fast if trading.jump_daily is absent or missing any column. This package never creates
    tables — the operator applies the DDL in Scala's database_schema.sql first."""
    present = {r[0] for r in ch.query(
        "SELECT name FROM system.columns WHERE database = 'trading' AND table = 'jump_daily'"
    ).result_rows}
    if not present:
        raise RuntimeError(
            f"{JUMP_DAILY_TABLE} does not exist. Apply its DDL from Scala's db/database_schema.sql "
            "first; this package never creates tables.")
    missing = [c for c in COLUMNS if c not in present]
    if missing:
        raise RuntimeError(
            f"{JUMP_DAILY_TABLE} is missing column(s): " + ", ".join(missing) + ". Apply the current "
            "DDL from Scala's db/database_schema.sql (this package never creates or alters tables).")


def _month_ranges(start: dt.date, end: dt.date):
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        nxt = dt.date(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
        yield max(cur, start), min(nxt - dt.timedelta(days=1), end)
        cur = nxt


def _insert_month(ch: Any, ms: dt.date, me: dt.date, symbols, only_missing: bool) -> int:
    """Run one month's set-based INSERT over [ms, me] (with a `_JUMP_LOOKBACK_DAYS` warm-up so the
    trailing medians are complete), then count and fail-fast on the month's rows. `only_missing` adds
    the `daily` filter (emit only rv_daily rows with no jump_daily row yet). Returns the row count."""
    lookback = ms - dt.timedelta(days=_JUMP_LOOKBACK_DAYS)
    lo, hi = lookback.isoformat(), me.isoformat()
    # SPY must be in the overnight recompute even on a targeted rerun; the rv_daily join carries the
    # emit restriction, so a targeted rerun still emits only the requested names.
    ov_sym = _symbol_in((list(symbols) + [SPY_SYMBOL]) if symbols else None)
    rv_sym = _symbol_in(symbols)
    only = (f" AND (symbol, trading_date) NOT IN "
            f"(SELECT symbol, trading_date FROM {JUMP_DAILY_TABLE} FINAL WHERE method = '{METHOD}')"
            if only_missing else "")

    jz = (f"if(bv_intraday IS NULL OR tq_intraday IS NULL OR rv_intraday = 0 OR bv_intraday = 0, NULL, "
          f"((rv_intraday - bv_intraday) / rv_intraday) "
          f"/ sqrt({THETA!r} * (1.0 / toFloat64(n_bars)) * greatest(1.0, tq_intraday / pow(bv_intraday, 2))))")
    oz = (f"if(n_prev < {ON_MIN_HISTORY} OR med_on IS NULL OR med_on = 0, NULL, "
          f"abs(overnight_ret) / sqrt(med_on))")
    oza = (f"if(overnight_ret_adj IS NULL OR n_prev_adj < {ON_MIN_HISTORY} OR med_on_adj IS NULL "
           f"OR med_on_adj = 0, NULL, abs(overnight_ret_adj) / sqrt(med_on_adj))")

    sql = f"""
INSERT INTO {JUMP_DAILY_TABLE}
    ({', '.join(COLUMNS)})
{_with_core(lookback, me, ov_sym)},
ov AS (
    SELECT symbol, trading_date, {OVERNIGHT_RET_SQL} AS overnight_ret
    FROM withsplit
    WHERE prev_close > 0 AND prev_complete_date = prev_trading_day
),
spy AS (
    SELECT trading_date, overnight_ret AS spy_ret FROM ov WHERE symbol = '{SPY_SYMBOL}'
),
rvd AS (
    SELECT symbol, trading_date, n_bars, rv_intraday, bv_intraday, tq_intraday, rv_overnight
    FROM {RV_DAILY_TABLE} FINAL
    WHERE source = '{RV_SOURCE}' AND trading_date >= toDate('{lo}') AND trading_date <= toDate('{hi}'){rv_sym}
),
base AS (
    SELECT rvd.symbol AS symbol, rvd.trading_date AS trading_date, rvd.n_bars AS n_bars,
           rvd.rv_intraday AS rv_intraday, rvd.bv_intraday AS bv_intraday, rvd.tq_intraday AS tq_intraday,
           rvd.rv_overnight AS rv_overnight, ov.overnight_ret AS overnight_ret,
           if(rvd.symbol = '{SPY_SYMBOL}', NULL, ov.overnight_ret - spy.spy_ret) AS overnight_ret_adj
    FROM rvd
    INNER JOIN ov ON rvd.symbol = ov.symbol AND rvd.trading_date = ov.trading_date
    LEFT JOIN spy ON rvd.trading_date = spy.trading_date
),
wn AS (
    SELECT *,
           quantileExactInclusive(0.5)(pow(overnight_ret, 2)) OVER w AS med_on,
           count() OVER w AS n_prev,
           quantileExactInclusive(0.5)(pow(overnight_ret_adj, 2)) OVER w AS med_on_adj,
           count(overnight_ret_adj) OVER w AS n_prev_adj
    FROM base
    WINDOW w AS (PARTITION BY symbol ORDER BY trading_date ROWS BETWEEN {ON_WINDOW} PRECEDING AND 1 PRECEDING)
),
z AS (
    SELECT *, {jz} AS jump_z, {oz} AS overnight_z, {oza} AS overnight_z_adj FROM wn
),
f AS (
    SELECT *,
           if(jump_z IS NULL, NULL, toUInt8(jump_z > {Z_INTRADAY!r})) AS intraday_jump,
           if(overnight_z IS NULL, NULL, toUInt8(overnight_z >= {Z_OVERNIGHT!r})) AS overnight_jump,
           if(overnight_z_adj IS NULL, NULL, toUInt8(overnight_z_adj >= {Z_OVERNIGHT!r})) AS overnight_jump_adj
    FROM z
),
jc AS (
    SELECT *,
           if(intraday_jump IS NULL, NULL,
              if(intraday_jump = 1, greatest(rv_intraday - bv_intraday, 0.0), 0.0)) AS jump_var
    FROM f
)
SELECT
    '{METHOD}' AS method, symbol, trading_date, n_bars, rv_intraday, bv_intraday,
    jump_z, intraday_jump, jump_var, rv_intraday - jump_var AS cont_var,
    overnight_ret, overnight_z, overnight_jump, overnight_ret_adj, overnight_z_adj, overnight_jump_adj,
    now64(3) AS computed_at
FROM jc
WHERE trading_date >= toDate('{ms.isoformat()}'){only}
SETTINGS join_use_nulls = 1
"""
    ch.command(sql)

    n = ch.query(
        f"SELECT count() FROM {JUMP_DAILY_TABLE} FINAL WHERE method = '{METHOD}' "
        f"AND trading_date >= '{ms.isoformat()}' AND trading_date <= '{me.isoformat()}'{rv_sym}"
    ).result_rows[0][0]
    _check_month(ch, ms, me, rv_sym, lo)
    log.info("jump_daily %s..%s: %d rows", ms, me, n)
    print(f"[jumps] {ms} .. {me}: {n:,} rows")
    return n


def _check_month(ch: Any, ms: dt.date, me: dt.date, rv_sym: str, lo: str) -> None:
    """Fail fast, by name and date: any non-finite value, a negative variance, cont_var+jump_var that
    doesn't equal rv_intraday (1e-12 + 1e-9·RV), or overnight_ret² that doesn't equal the stored
    rv_overnight (1e-12 + 1e-6·rv_overnight). Then the lookback-truncation guard — a median must never
    be taken over fewer than a row's true ON_WINDOW history because the month's lookback was too short."""
    jsym = (" AND j.symbol IN (" + rv_sym.split("IN (", 1)[1]) if rv_sym else ""
    bad = ch.query(
        f"SELECT j.symbol, toString(j.trading_date), j.jump_z, j.jump_var, j.cont_var, j.overnight_z, "
        f"j.overnight_ret, j.overnight_z_adj, j.overnight_ret_adj, j.rv_intraday, r.rv_overnight "
        f"FROM {JUMP_DAILY_TABLE} j FINAL "
        f"INNER JOIN (SELECT symbol, trading_date, rv_overnight FROM {RV_DAILY_TABLE} FINAL "
        f"            WHERE source = '{RV_SOURCE}') r "
        f"  ON j.symbol = r.symbol AND j.trading_date = r.trading_date "
        f"WHERE j.method = '{METHOD}' AND j.trading_date >= '{ms.isoformat()}' "
        f"AND j.trading_date <= '{me.isoformat()}'{jsym} AND ("
        f"  (j.jump_z IS NOT NULL AND NOT isFinite(j.jump_z))"
        f"  OR (j.jump_var IS NOT NULL AND (NOT isFinite(j.jump_var) OR j.jump_var < 0))"
        f"  OR (j.cont_var IS NOT NULL AND (NOT isFinite(j.cont_var) OR j.cont_var < 0))"
        f"  OR (j.overnight_z IS NOT NULL AND NOT isFinite(j.overnight_z))"
        f"  OR NOT isFinite(j.overnight_ret)"
        f"  OR (j.overnight_z_adj IS NOT NULL AND NOT isFinite(j.overnight_z_adj))"
        f"  OR (j.overnight_ret_adj IS NOT NULL AND NOT isFinite(j.overnight_ret_adj))"
        f"  OR (j.jump_var IS NOT NULL AND abs(j.cont_var + j.jump_var - j.rv_intraday) > 1e-12 + 1e-9 * j.rv_intraday)"
        f"  OR (abs(pow(j.overnight_ret, 2) - r.rv_overnight) > 1e-12 + 1e-6 * r.rv_overnight)"
        f") ORDER BY j.symbol, j.trading_date LIMIT 100"
    ).result_rows
    if bad:
        lines = "\n".join(
            f"    {s:8s} {d}  jump_z={jz} jump_var={jv} cont_var={cv} overnight_z={oz} "
            f"overnight_ret={orr} rv_intraday={rvi} rv_overnight={rvo}"
            for s, d, jz, jv, cv, oz, orr, oza_, ora, rvi, rvo in bad)
        raise RuntimeError(
            f"jump_daily produced invalid rows for {ms}..{me} ({len(bad)}+ shown):\n{lines}")

    # Truncation guard: for each emitted row, how many of its true 66 preceding rv_daily rows fell
    # inside this month's lookback (look_prev) vs the size of that true window (frame66). If any row
    # saw fewer than frame66, the lookback was too short and that row's median was taken over fewer
    # sessions than the rule requires — fail and say to raise _JUMP_LOOKBACK_DAYS. Reads rv_daily only.
    trunc = ch.query(
        f"SELECT symbol, toString(trading_date), look_prev, frame66 FROM ("
        f"  SELECT symbol, trading_date, "
        f"    sum(trading_date >= toDate('{lo}')) OVER w66 AS look_prev, count() OVER w66 AS frame66 "
        f"  FROM {RV_DAILY_TABLE} FINAL WHERE source = '{RV_SOURCE}'{rv_sym} "
        f"  WINDOW w66 AS (PARTITION BY symbol ORDER BY trading_date "
        f"                 ROWS BETWEEN {ON_WINDOW} PRECEDING AND 1 PRECEDING)) "
        f"WHERE trading_date >= '{ms.isoformat()}' AND trading_date <= '{me.isoformat()}' "
        f"AND look_prev < frame66 ORDER BY symbol, trading_date LIMIT 100"
    ).result_rows
    if trunc:
        lines = "\n".join(f"    {s:8s} {d}  saw {lp} of {f6} preceding sessions" for s, d, lp, f6 in trunc)
        raise RuntimeError(
            f"jump_daily lookback too short for {ms}..{me}: {len(trunc)}+ rows whose trailing median "
            f"would be taken over fewer than their ON_WINDOW history — raise _JUMP_LOOKBACK_DAYS "
            f"(currently {_JUMP_LOOKBACK_DAYS} days):\n{lines}")


def backfill(ch: Any, start: dt.date, end: dt.date, symbols=None) -> int:
    """Compute jump_daily for every rv_daily row in [start, end], one statement per month.
    Re-insert-safe; `symbols` restricts the emitted names (SPY is always recomputed for the adjustment)."""
    _require_table(ch)
    total = 0
    for ms, me in _month_ranges(start, end):
        total += _insert_month(ch, ms, me, symbols, only_missing=False)
    return total


def daily(ch: Any) -> int:
    """Compute every rv_daily row (source har-rv) that has no jump_daily row yet, from the earliest
    such day to the latest rv_daily day, one statement per month (with lookback). Prints `wrote N
    rows` for the Scala JumpJob to parse. Data-driven like rv_daily.incremental."""
    _require_table(ch)
    start = ch.query(
        f"SELECT minOrNull(trading_date) FROM {RV_DAILY_TABLE} FINAL WHERE source = '{RV_SOURCE}' "
        f"AND (symbol, trading_date) NOT IN (SELECT symbol, trading_date FROM {JUMP_DAILY_TABLE} FINAL "
        f"WHERE method = '{METHOD}')"
    ).result_rows[0][0]
    if start is None or start <= dt.date(1970, 1, 1):
        print("wrote 0 rows (no rv_daily rows without a jump_daily row)")
        return 0
    end = ch.query(
        f"SELECT maxOrNull(trading_date) FROM {RV_DAILY_TABLE} FINAL WHERE source = '{RV_SOURCE}'"
    ).result_rows[0][0]
    print(f"[jumps] daily {start} .. {end} (every rv_daily row with no jump_daily row)")
    total = 0
    for ms, me in _month_ranges(start, end):
        total += _insert_month(ch, ms, me, None, only_missing=True)
    print(f"wrote {total} rows")
    return total


# ── backfill summary (printed once at the end — the proof the measures work) ──────────────────────

def summary_flag_shares_by_year(ch: Any) -> List[tuple]:
    """Per year: (year, n_ij, ij, n_oj, oj, n_oja, oja) — the non-null denominators and the =1 counts
    for intraday_jump, overnight_jump, overnight_jump_adj."""
    return ch.query(
        f"SELECT toYear(trading_date) AS y, "
        f"countIf(intraday_jump IS NOT NULL), countIf(intraday_jump = 1), "
        f"countIf(overnight_jump IS NOT NULL), countIf(overnight_jump = 1), "
        f"countIf(overnight_jump_adj IS NOT NULL), countIf(overnight_jump_adj = 1) "
        f"FROM {JUMP_DAILY_TABLE} FINAL WHERE method = '{METHOD}' GROUP BY y ORDER BY y"
    ).result_rows


def summary_top_overnight_dates(ch: Any, limit: int = 15) -> List[tuple]:
    """The dates with the most overnight_jump names, each beside its overnight_jump_adj count."""
    return ch.query(
        f"SELECT toString(trading_date), countIf(overnight_jump = 1) AS raw, "
        f"countIf(overnight_jump_adj = 1) AS adj "
        f"FROM {JUMP_DAILY_TABLE} FINAL WHERE method = '{METHOD}' "
        f"GROUP BY trading_date ORDER BY raw DESC, trading_date LIMIT {limit}"
    ).result_rows


def summary_earnings_explains(ch: Any) -> tuple:
    """(total, explained, share) — of overnight_jump_adj = 1 rows, how many have an earnings_history
    report on that day or the previous trading day. Read-only."""
    rows = ch.query(
        f"SELECT symbol, trading_date FROM {JUMP_DAILY_TABLE} FINAL "
        f"WHERE method = '{METHOD}' AND overnight_jump_adj = 1"
    ).result_rows
    total = len(rows)
    if total == 0:
        return (0, 0, 0.0)
    syms = sorted({str(s) for s, _ in rows})
    quoted = ", ".join("'" + s.replace("'", "") + "'" for s in syms)
    reported: dict[str, set] = {}
    for sym, rd in ch.query(
        f"SELECT symbol, reported_date FROM {EARNINGS_HISTORY_TABLE} FINAL "
        f"WHERE symbol IN ({quoted})"
    ).result_rows:
        reported.setdefault(str(sym), set()).add(rd)
    explained = sum(
        1 for sym, d in rows
        if d in reported.get(str(sym), set()) or previous_trading_day(d) in reported.get(str(sym), set()))
    return (total, explained, explained / total)


def summary_orphans(ch: Any, limit: int = 20) -> tuple:
    """(count, [up to `limit` (symbol, date)]) — jump_daily rows with no rv_daily row. Reports only."""
    where = (f"FROM {JUMP_DAILY_TABLE} FINAL WHERE method = '{METHOD}' "
             f"AND (symbol, trading_date) NOT IN "
             f"(SELECT symbol, trading_date FROM {RV_DAILY_TABLE} FINAL WHERE source = '{RV_SOURCE}')")
    count = ch.query(f"SELECT count() {where}").result_rows[0][0]
    listed = ch.query(
        f"SELECT symbol, toString(trading_date) {where} ORDER BY symbol, trading_date LIMIT {limit}"
    ).result_rows
    return (count, listed)


def print_backfill_summary(ch: Any) -> None:
    print("\n[jumps] ── backfill summary ──")
    print("[jumps] 1. flag shares by year (=1 / non-null):")
    print(f"    {'year':6s} {'intraday':>18s} {'overnight':>18s} {'overnight_adj':>18s}")
    for y, n_ij, ij, n_oj, oj, n_oja, oja in summary_flag_shares_by_year(ch):
        def sh(num, den):
            return f"{(num / den * 100 if den else 0):.3f}% ({num}/{den})"
        print(f"    {y:<6d} {sh(ij, n_ij):>18s} {sh(oj, n_oj):>18s} {sh(oja, n_oja):>18s}")

    print("[jumps] 2. top 15 dates by overnight_jump names (raw vs SPY-adjusted):")
    print(f"    {'date':12s} {'overnight':>10s} {'overnight_adj':>14s}")
    for d, raw, adj in summary_top_overnight_dates(ch):
        print(f"    {d:12s} {raw:>10d} {adj:>14d}")

    total, explained, share = summary_earnings_explains(ch)
    print(f"[jumps] 3. overnight_jump_adj=1 rows explained by own earnings (report day or previous "
          f"trading day): {explained}/{total} = {share * 100:.1f}%")

    count, listed = summary_orphans(ch)
    print(f"[jumps] 4. orphan jump_daily rows (no rv_daily row): {count:,}")
    for s, d in listed:
        print(f"    {s:8s} {d}")


# ── read-only inspection ──────────────────────────────────────────────────────────────────────────

def _f(x, fmt: str = "{:.3f}") -> str:
    return "—" if x is None else fmt.format(float(x))


def show(ch: Any, symbol: str, frm: dt.date, to: dt.date) -> None:
    """One line per day for `symbol` over [frm, to]: date, n_bars, jump_z, intraday flag, jump share
    of RV, overnight_ret, overnight_z, overnight_ret_adj, overnight_z_adj, flags. Read-only."""
    rows = ch.query(
        f"SELECT toString(trading_date), n_bars, jump_z, intraday_jump, "
        f"if(rv_intraday > 0, jump_var / rv_intraday, NULL) AS jump_share, "
        f"overnight_ret, overnight_z, overnight_ret_adj, overnight_z_adj, overnight_jump, overnight_jump_adj "
        f"FROM {JUMP_DAILY_TABLE} FINAL WHERE method = '{METHOD}' AND symbol = '{symbol.upper()}' "
        f"AND trading_date >= '{frm.isoformat()}' AND trading_date <= '{to.isoformat()}' ORDER BY trading_date"
    ).result_rows
    print(f"jump_daily {METHOD}  {symbol.upper()}  {frm} .. {to}")
    if not rows:
        print("  (no rows)")
        return
    print(f"    {'date':12s} {'n':>4s} {'jump_z':>8s} {'ij':>3s} {'jumpRV':>7s} "
          f"{'ovn_ret':>9s} {'ovn_z':>7s} {'ovn_adj':>9s} {'ovn_z_adj':>9s} {'oj':>3s} {'oja':>4s}")
    for d, n, jz, ij, jsh, orr, oz, ora, oza, oj, oja in rows:
        print(f"    {d:12s} {int(n):>4d} {_f(jz, '{:+.2f}'):>8s} {_f(ij, '{:.0f}'):>3s} "
              f"{_f(jsh, '{:.3f}'):>7s} {_f(orr, '{:+.4f}'):>9s} {_f(oz, '{:.2f}'):>7s} "
              f"{_f(ora, '{:+.4f}'):>9s} {_f(oza, '{:.2f}'):>9s} {_f(oj, '{:.0f}'):>3s} {_f(oja, '{:.0f}'):>4s}")
