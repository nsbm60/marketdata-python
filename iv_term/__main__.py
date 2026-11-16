"""
Daily ATM term-structure snapshot commands (pc-objective-function-draft.md §6–§7).

    python -m iv_term backfill <from> <to>   # one calendar month per statement, from..to inclusive
    python -m iv_term daily                  # newest ledger day (incl) .. newest option_trade day
    python -m iv_term show <date> <underlying>

This package never creates tables: the trading.iv_term_daily DDL lives in Scala's database_schema.sql
and is applied by hand.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import signal
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.shared.clickhouse import get_ch_client  # noqa: E402
from iv_term import show as show_mod  # noqa: E402
from iv_term import term  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(prog="iv_term")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill")
    b.add_argument("start", metavar="from", help="first session_date, YYYY-MM-DD (inclusive)")
    b.add_argument("end", metavar="to", help="last session_date, YYYY-MM-DD (inclusive)")
    sub.add_parser("daily")
    s = sub.add_parser("show")
    s.add_argument("date", help="session_date, YYYY-MM-DD")
    s.add_argument("underlying", help="underlying ticker")
    args = p.parse_args()

    ch: Any = get_ch_client()

    if args.cmd == "backfill":
        start = dt.date.fromisoformat(args.start)
        end = dt.date.fromisoformat(args.end)
        # Clean stop: a SIGINT sets the flag; backfill finishes the current month, then stops.
        signal.signal(signal.SIGINT, lambda *_: term.request_stop())
        print(f"[iv_term] backfill {start} .. {end}")
        total = term.backfill(ch, get_ch_client, start, end)
        print(f"[iv_term] backfill done: {term._summary(total)}")
    elif args.cmd == "daily":
        n = term.daily(ch)
        if n < 0:
            sys.exit(1)
    elif args.cmd == "show":
        show_mod.show(ch, dt.date.fromisoformat(args.date), args.underlying)


if __name__ == "__main__":
    main()
