"""CLI argument parsing and excluded-date guard (no network)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from greeks.config import load_config
from greeks.pull.run import build_parser, ensure_session_allowed


def test_parser_dates() -> None:
    p = build_parser()
    args = p.parse_args(
        ["--ticker", "NVDA", "--date", "2026-05-27", "--max-contracts", "3"]
    )
    assert args.ticker == "NVDA"
    assert args.date == date(2026, 5, 27)
    assert args.max_contracts == 3


def test_excluded_date_refused() -> None:
    cfg = load_config()
    with pytest.raises(SystemExit):
        ensure_session_allowed(cfg, date(2026, 6, 8))


def test_in_window_ok() -> None:
    cfg = load_config()
    ensure_session_allowed(cfg, date(2026, 5, 27))


def test_outside_window_refused() -> None:
    cfg = load_config()
    with pytest.raises(SystemExit):
        ensure_session_allowed(cfg, date(2020, 1, 1))
