"""
Read-only data-quality pass over trading.rv_daily (HAR edge). Writes nothing.

    cd ~/Projects/Python/marketdata-python && set -a && source .env && set +a && venv/bin/python tools/rv_dq.py

Sections:
  A. The 20 largest rv_total days, with intraday/overnight split, the day's open and the prior
     complete close (recomputed from stock_bar), and split_adjusted.
  B. Splits on these names since 2016 with no split_adjusted rv_daily row, with the reason.
  C. Every zero-RV day, with bar count vs expected.
  D. SPAC scan: for each zero-RV name, the zero-day date range and the open on the first zero day
     (a SPAC shell sits near $10); plus VRT daily closes across the 2020 merger window.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.shared.clickhouse import get_ch_client  # noqa: E402


def _scalar(ch, sql):
    r = ch.query(sql).result_rows
    return r[0][0] if r and r[0] and r[0][0] is not None else None


def section_a(ch):
    print("=== A. 20 largest rv_total days ===")
    top = ch.query(
        "SELECT symbol, toString(trading_date), rv_intraday, rv_overnight, rv_total, split_adjusted "
        "FROM trading.rv_daily FINAL ORDER BY rv_total DESC LIMIT 20"
    ).result_rows
    print(f"{'symbol':8s} {'date':12s} {'rv_intraday':>12s} {'rv_overnight':>12s} "
          f"{'open':>10s} {'prev_close':>11s} {'split':>5s}")
    for sym, d, rv_i, rv_o, rv_t, adj in top:
        day_open = _scalar(ch, f"SELECT argMin(open, ts) FROM trading.stock_bar WHERE symbol='{sym}' "
                                f"AND period='1m' AND session=1 AND trading_date='{d}'")
        prev_date = _scalar(ch, f"SELECT max(trading_date) FROM trading.stock_bar_ledger FINAL "
                                 f"WHERE symbol='{sym}' AND status='complete' AND trading_date<'{d}'")
        prev_close = None
        if prev_date is not None:
            prev_close = _scalar(ch, f"SELECT argMax(close, ts) FROM trading.stock_bar WHERE symbol='{sym}' "
                                      f"AND period='1m' AND session=1 AND trading_date='{prev_date}'")
        o = f"{day_open:.2f}" if day_open is not None else "—"
        pc = f"{prev_close:.2f}" if prev_close is not None else "—"
        print(f"{sym:8s} {d:12s} {rv_i:12.3e} {rv_o:12.3e} {o:>10s} {pc:>11s} {adj:>5d}")


def section_b(ch):
    print("\n=== B. splits since 2016 with no split_adjusted rv_daily row ===")
    missing = ch.query(
        "SELECT s.symbol, toString(s.execution_date) AS d, s.ratio FROM "
        "(SELECT symbol, execution_date, any(ratio) AS ratio FROM trading.split "
        " WHERE execution_date >= '2016-01-01' "
        "   AND symbol IN (SELECT DISTINCT symbol FROM trading.rv_daily) "
        " GROUP BY symbol, execution_date) s "
        "WHERE (s.symbol, s.execution_date) NOT IN "
        "(SELECT symbol, trading_date FROM trading.rv_daily FINAL WHERE split_adjusted = 1) "
        "ORDER BY s.symbol, s.execution_date"
    ).result_rows
    print(f"{'symbol':8s} {'exec_date':12s} {'ratio':>8s}  reason")
    for sym, d, ratio in missing:
        status = _scalar(ch, f"SELECT status FROM trading.stock_bar_ledger FINAL "
                              f"WHERE symbol='{sym}' AND trading_date='{d}'")
        first_complete = _scalar(ch, f"SELECT min(trading_date) FROM trading.stock_bar_ledger FINAL "
                                      f"WHERE symbol='{sym}' AND status='complete'")
        has_rv = _scalar(ch, f"SELECT count() FROM trading.rv_daily FINAL "
                              f"WHERE symbol='{sym}' AND trading_date='{d}'")
        if status is None:
            reason = "not a trading day for this name (no ledger row)"
        elif first_complete is not None and d < str(first_complete):
            reason = f"before listing (first complete {first_complete})"
        elif status != "complete":
            reason = f"ledger status '{status}' (not a complete day)"
        elif has_rv == 0:
            reason = "complete day but gap-dropped (overnight would span a gap)"
        else:
            reason = "other — rv_daily row exists but split_adjusted=0 (investigate)"
        print(f"{sym:8s} {d:12s} {ratio:>8.4f}  {reason}")


def section_c(ch):
    print("\n=== C. zero-RV days (bar count vs expected) ===")
    zeros = ch.query(
        "SELECT symbol, toString(trading_date), n_bars, expected_bars FROM trading.rv_daily FINAL "
        "WHERE rv_total = 0 ORDER BY symbol, trading_date"
    ).result_rows
    print(f"total {len(zeros)} zero-RV days")
    print(f"{'symbol':8s} {'date':12s} {'n_bars':>7s} {'expected':>9s}")
    for sym, d, n, exp in zeros:
        print(f"{sym:8s} {d:12s} {n:>7d} {exp:>9d}")


def section_d(ch):
    print("\n=== D. SPAC scan — zero-RV names (a SPAC shell sits near $10) ===")
    names = ch.query(
        "SELECT symbol, count() AS n, toString(min(trading_date)) AS lo, toString(max(trading_date)) AS hi "
        "FROM trading.rv_daily FINAL WHERE rv_total = 0 GROUP BY symbol ORDER BY n DESC, symbol"
    ).result_rows
    print(f"{'symbol':8s} {'zeros':>5s} {'first_zero':12s} {'last_zero':12s} {'open@first':>10s}")
    for sym, n, lo, hi, in [(r[0], r[1], r[2], r[3]) for r in names]:
        first_open = _scalar(ch, f"SELECT argMin(open, ts) FROM trading.stock_bar WHERE symbol='{sym}' "
                                  f"AND period='1m' AND session=1 AND trading_date='{lo}'")
        o = f"{first_open:.2f}" if first_open is not None else "—"
        print(f"{sym:8s} {n:>5d} {lo:12s} {hi:12s} {o:>10s}")

    print("\n--- VRT daily closes across the 2020 merger window ---")
    rows = ch.query(
        "SELECT toString(trading_date) AS d, argMin(open, ts) AS o, argMax(close, ts) AS c, "
        "       round(stddevPop(close) / avg(close), 5) AS intraday_cv "
        "FROM trading.stock_bar WHERE symbol='VRT' AND period='1m' AND session=1 "
        "AND trading_date BETWEEN '2020-01-02' AND '2020-03-06' "
        "GROUP BY trading_date ORDER BY trading_date"
    ).result_rows
    print(f"{'date':12s} {'open':>8s} {'close':>8s} {'intraday_cv':>12s}")
    for d, o, c, cv in rows:
        print(f"{d:12s} {o:>8.2f} {c:>8.2f} {cv:>12.5f}")


def main():
    ch = get_ch_client()
    section_a(ch)
    section_b(ch)
    section_c(ch)
    section_d(ch)


if __name__ == "__main__":
    main()
