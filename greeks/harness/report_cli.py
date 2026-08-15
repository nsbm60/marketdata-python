"""CLI: print acceptance report from local results DB.

Usage::

    python -m greeks.harness.report_cli --results-db data/greeks_results.db
    python -m greeks.harness.report_cli --earnings NVDA:2026-05-28,MU:2026-06-18
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from greeks.config import load_config
from greeks.harness.report import build_report, format_report
from greeks.harness.store import ResultsStore


def _parse_earnings(s: str) -> list[tuple[str, date]]:
    """``NVDA:2026-05-28,MU:2026-06-18``."""
    out: list[tuple[str, date]] = []
    if not s.strip():
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        und, d = part.split(":", 1)
        out.append((und.strip().upper(), date.fromisoformat(d.strip())))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Greeks validation acceptance report (PR6)")
    p.add_argument(
        "--results-db",
        type=Path,
        default=Path("data/greeks_results.db"),
    )
    p.add_argument(
        "--methodology-version",
        type=str,
        default=None,
        help="Default: from greeks.yaml",
    )
    p.add_argument(
        "--earnings",
        type=str,
        default="",
        help="Optional earnings events UNDERLYING:YYYY-MM-DD,...",
    )
    p.add_argument("--config", type=Path, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    mv = args.methodology_version or cfg.methodology_version

    with ResultsStore(args.results_db) as store:
        validation = store.load_validation(methodology_version=mv)
        residuals = store.load_residuals(methodology_version=mv)

    if not validation and not residuals:
        raise SystemExit(f"no rows in {args.results_db} for methodology={mv}")

    report = build_report(
        methodology_version=mv,
        validation=validation,
        residuals=residuals,
        earnings_events=_parse_earnings(args.earnings),
    )
    print(format_report(report))
    sys.exit(0 if report.overall_pass else 1)


if __name__ == "__main__":
    main()
