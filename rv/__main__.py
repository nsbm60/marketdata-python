"""
HAR realized-variance commands (HAR edge spec §3–§5).

    python -m rv backfill [--start 2016-01-01] [--end YYYY-MM-DD]   # rv_daily from raw 1m bars
    python -m rv proof                                              # AAPL/NVDA split sanity
    python -m rv forecast                                           # pooled fit + historical forecasts
    python -m rv all      [--start ...] [--end ...]                 # backfill, proof, forecast

The walk-forward report (§6) is a separate script that writes nothing: tools/rv_eval.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.shared.clickhouse import get_ch_client  # noqa: E402
from rv import forecast, rv_daily  # noqa: E402
from rv.schema import ensure_tables  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _dates(args) -> tuple[dt.date, dt.date]:
    start = dt.date.fromisoformat(args.start) if args.start else dt.date(2016, 1, 1)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today() - dt.timedelta(days=1)
    return start, end


def _run_proof(ch):
    rows = rv_daily.split_proof(ch)
    print("[proof] split handling — overnight move should be ordinary, not split-sized:")
    print(f"    {'symbol':8s} {'date':12s} {'rv_overnight':>14s} {'|ovn ret|':>10s} {'rv_total':>12s} {'split_adj':>9s}")
    for sym, d, rv_ovn, ovn_abs, rv_intr, rv_tot, adj in rows:
        print(f"    {sym:8s} {d:12s} {rv_ovn:14.3e} {ovn_abs:10.4f} {rv_tot:12.3e} {adj:>9d}")


def main() -> None:
    p = argparse.ArgumentParser(prog="rv")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("backfill", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("--start")
        sp.add_argument("--end")
    sub.add_parser("proof")
    sub.add_parser("forecast")
    args = p.parse_args()

    ch = get_ch_client()
    ensure_tables(ch)

    if args.cmd in ("backfill", "all"):
        start, end = _dates(args)
        print(f"[backfill] rv_daily {start} .. {end}")
        total = rv_daily.backfill(ch, start, end)
        dropped = rv_daily.gap_dropped_count(ch)
        st = rv_daily.stats(ch)
        print(f"[backfill] {total:,} rv_daily rows")
        print(f"[backfill] dropped {dropped:,} complete days whose previous complete day isn't the "
              f"previous trading day (overnight would span a gap)")
        print(f"[backfill] split_adjusted=1 on {st['split_adjusted']:,} rows "
              f"({st['splits_in_table']:,} splits on these names since 2016)")
        print(f"[backfill] rv_total  min={st['rv_min']:.3e}  median={st['rv_median']:.3e}  max={st['rv_max']:.3e}")
        zeros = rv_daily.zero_rv_by_name(ch)
        total_zero = sum(n for _, n in zeros)
        print(f"[backfill] zero-RV days: {total_zero:,} across {len(zeros)} names (stored, excluded from the fit)")
        for sym, n in zeros:
            print(f"    {sym:8s} {n}")
    if args.cmd in ("proof", "all"):
        _run_proof(ch)
    if args.cmd in ("forecast", "all"):
        forecast.run(ch)


if __name__ == "__main__":
    main()
