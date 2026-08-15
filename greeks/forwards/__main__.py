"""CLI: SOFR load helpers.

Usage::

    # Push package fixture into trading.sofr_daily
    python -m greeks.forwards --load-fixture

    # FRED pull for a date range then insert (needs FRED_API_KEY)
    python -m greeks.forwards --fred --start 2026-05-01 --end 2026-07-20
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from typing import Optional

from greeks.ch import get_ch_client, insert_sofr_observations, load_sofr_fixture_to_ch
from greeks.forwards.sofr import fetch_fred_sofr


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SOFR → ClickHouse loaders")
    p.add_argument(
        "--load-fixture",
        action="store_true",
        help="Insert package SOFR CSV fixture into trading.sofr_daily",
    )
    p.add_argument(
        "--fred",
        action="store_true",
        help="Pull FRED SOFR and insert (requires FRED_API_KEY)",
    )
    p.add_argument("--start", type=_parse_date, default=date(2026, 5, 1))
    p.add_argument("--end", type=_parse_date, default=date(2026, 7, 20))
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV path for --load-fixture",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.load_fixture and not args.fred:
        raise SystemExit("specify --load-fixture and/or --fred")

    client = get_ch_client()
    if args.load_fixture:
        n = load_sofr_fixture_to_ch(client, csv_path=args.csv)
        print(f"fixture: inserted {n} rows into trading.sofr_daily")
    if args.fred:
        key = os.environ.get("FRED_API_KEY", "")
        if not key:
            raise SystemExit("FRED_API_KEY not set")
        obs = fetch_fred_sofr(
            key,
            observation_start=args.start,
            observation_end=args.end,
        )
        n = insert_sofr_observations(client, obs)
        print(f"fred: inserted {n} rows ({args.start}..{args.end})")


if __name__ == "__main__":
    main()
