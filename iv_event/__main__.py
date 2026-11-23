"""
Implied event detector commands (roadmap-2026-11-16 Stage 3, method fwd_v1).

    python -m iv_event daytype <from> <to>        # measure & write the day weights, print the table
    python -m iv_event backfill <from> <to>       # one calendar month per statement, from..to inclusive
    python -m iv_event daily                      # windows, then outcomes for newly-passed windows
    python -m iv_event show <date> <underlying>
    python -m iv_event top <from> <to> [--min-support N]
    python -m iv_event outcome-backfill <from> <to>   # event outcomes for passed windows, session_date from..to
    python -m iv_event outcome-show <date> <underlying>

This package never creates tables: the trading.iv_event_window, trading.iv_event_daytype and
trading.iv_event_outcome DDL lives in Scala's database_schema.sql and is applied by hand.
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
from iv_event import (  # noqa: E402
    TOP_DEFAULT_MIN_SUPPORT, daytype as daytype_mod, event, outcome_event, show as show_mod,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(prog="iv_event")
    sub = p.add_subparsers(dest="cmd", required=True)
    dayt = sub.add_parser("daytype")
    dayt.add_argument("start", metavar="from", help="first trading_date, YYYY-MM-DD (inclusive)")
    dayt.add_argument("end", metavar="to", help="last trading_date, YYYY-MM-DD (inclusive)")
    b = sub.add_parser("backfill")
    b.add_argument("start", metavar="from", help="first session_date, YYYY-MM-DD (inclusive)")
    b.add_argument("end", metavar="to", help="last session_date, YYYY-MM-DD (inclusive)")
    sub.add_parser("daily")
    s = sub.add_parser("show")
    s.add_argument("date", help="session_date, YYYY-MM-DD")
    s.add_argument("underlying", help="underlying ticker")
    t = sub.add_parser("top")
    t.add_argument("start", metavar="from", help="first session_date, YYYY-MM-DD (inclusive)")
    t.add_argument("end", metavar="to", help="last session_date, YYYY-MM-DD (inclusive)")
    t.add_argument("--min-support", type=float, default=TOP_DEFAULT_MIN_SUPPORT,
                   help=f"minimum window support (default {TOP_DEFAULT_MIN_SUPPORT:g})")
    ob = sub.add_parser("outcome-backfill")
    ob.add_argument("start", metavar="from", help="first session_date, YYYY-MM-DD (inclusive)")
    ob.add_argument("end", metavar="to", help="last session_date, YYYY-MM-DD (inclusive)")
    os_ = sub.add_parser("outcome-show")
    os_.add_argument("date", help="session_date, YYYY-MM-DD")
    os_.add_argument("underlying", help="underlying ticker")
    args = p.parse_args()

    ch: Any = get_ch_client()

    if args.cmd == "daytype":
        start = dt.date.fromisoformat(args.start)
        end = dt.date.fromisoformat(args.end)
        rows = daytype_mod.run(ch, start, end)
        daytype_mod.print_table(rows, start, end)
    elif args.cmd == "backfill":
        start = dt.date.fromisoformat(args.start)
        end = dt.date.fromisoformat(args.end)
        # Clean stop: a SIGINT sets the flag; backfill finishes the current month, then stops.
        signal.signal(signal.SIGINT, lambda *_: event.request_stop())
        print(f"[iv_event] backfill {start} .. {end}")
        try:
            total = event.backfill(ch, get_ch_client, start, end)
        except event.DaytypeMissing as exc:
            print(str(exc))
            sys.exit(1)
        print(f"[iv_event] backfill done: {event._summary(total)}")
    elif args.cmd == "daily":
        n = event.daily(ch)
        if n < 0:
            sys.exit(1)
        # After the windows, compute outcomes for every passed window with no out_v1 row yet. Its
        # summary line does not match the Scala IvEventJob `wrote N rows` regex, so the parsed window
        # count is unchanged. A missing outcome table is a hard prerequisite → non-zero exit.
        try:
            if outcome_event.outcome_daily(ch) < 0:
                sys.exit(1)
        except outcome_event.OutcomeTableMissing as exc:
            print(str(exc))
            sys.exit(1)
    elif args.cmd == "show":
        show_mod.show(ch, dt.date.fromisoformat(args.date), args.underlying)
    elif args.cmd == "top":
        show_mod.top(ch, dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end),
                     args.min_support)
    elif args.cmd == "outcome-backfill":
        start = dt.date.fromisoformat(args.start)
        end = dt.date.fromisoformat(args.end)
        signal.signal(signal.SIGINT, lambda *_: event.request_stop())
        try:
            n_outcomes = outcome_event.outcome_backfill(ch, get_ch_client, start, end)
        except outcome_event.OutcomeTableMissing as exc:
            print(str(exc))
            sys.exit(1)
        if n_outcomes < 0:
            sys.exit(1)
        outcome_event.print_backfill_summary(ch)
    elif args.cmd == "outcome-show":
        show_mod.outcome_show(ch, dt.date.fromisoformat(args.date), args.underlying)


if __name__ == "__main__":
    main()
