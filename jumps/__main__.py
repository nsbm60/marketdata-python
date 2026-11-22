"""
Jump-day record commands (roadmap-2026-11-16 Stage 3b part 1, method bns_v1).

    python -m jumps backfill [--start 2016-01-01] [--end YYYY-MM-DD] [--symbols A,B]
    python -m jumps daily                         # every rv_daily row with no jump_daily row yet
    python -m jumps show SYMBOL FROM TO            # one line per day (read-only)

This package never creates tables: the trading.jump_daily DDL lives in Scala's database_schema.sql
and is applied by hand.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.shared.clickhouse import get_ch_client  # noqa: E402
from jumps import jump_daily  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(prog="jumps")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill")
    b.add_argument("--start")
    b.add_argument("--end")
    b.add_argument("--symbols", help="comma-separated names for a targeted rerun (default: all)")
    sub.add_parser("daily")
    s = sub.add_parser("show")
    s.add_argument("symbol")
    s.add_argument("start", metavar="from", help="first trading_date, YYYY-MM-DD (inclusive)")
    s.add_argument("end", metavar="to", help="last trading_date, YYYY-MM-DD (inclusive)")
    args = p.parse_args()

    ch: Any = get_ch_client()

    if args.cmd == "backfill":
        start = dt.date.fromisoformat(args.start) if args.start else dt.date(2016, 1, 1)
        end = dt.date.fromisoformat(args.end) if args.end else dt.date.today() - dt.timedelta(days=1)
        symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
        print(f"[jumps] backfill {start} .. {end}" + (f" symbols={symbols}" if symbols else ""))
        total = jump_daily.backfill(ch, start, end, symbols)
        print(f"[jumps] {total:,} jump_daily rows")
        jump_daily.print_backfill_summary(ch)
    elif args.cmd == "daily":
        jump_daily.daily(ch)
    elif args.cmd == "show":
        jump_daily.show(ch, args.symbol, dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end))


if __name__ == "__main__":
    main()
