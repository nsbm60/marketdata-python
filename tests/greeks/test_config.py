"""PR0: config chokepoint loads defaults and validates window exclusions."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from greeks.config import get_config, load_config
from greeks.domain import Dividend


def test_load_default_config() -> None:
    get_config.cache_clear()
    cfg = load_config()
    assert cfg.methodology_version == "v1"
    assert cfg.window.start == date(2026, 5, 27)
    assert cfg.window.end == date(2026, 7, 9)
    assert date(2026, 6, 8) in cfg.window.excluded_dates
    assert "NVDA" in cfg.tickers
    assert "MU" in cfg.tickers
    assert cfg.t_floor_minutes == 15
    assert cfg.join_staleness_s == 300
    assert cfg.day_count == "ACT/365"


def test_excluded_dates_list_not_single_hardcode() -> None:
    cfg = load_config()
    # Must be a sequence we can extend later without code changes.
    assert isinstance(cfg.window.excluded_dates, tuple)
    assert len(cfg.window.excluded_dates) >= 1


def test_window_contains() -> None:
    cfg = load_config()
    assert cfg.window.contains(date(2026, 5, 27)) is True
    assert cfg.window.contains(date(2026, 6, 8)) is False  # excluded
    assert cfg.window.contains(date(2026, 5, 1)) is False  # before window


def test_nvda_and_mu_dividends() -> None:
    cfg = load_config()
    divs = cfg.dividends_for("NVDA")
    assert divs == (
        Dividend(underlying="NVDA", amount=0.25, ex_date=date(2026, 6, 4)),
    )
    assert cfg.dividends_for("MU") == (
        Dividend(underlying="MU", amount=0.15, ex_date=date(2026, 7, 6)),
    )


def test_missing_config_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)


def test_get_config_cached() -> None:
    get_config.cache_clear()
    a = get_config()
    b = get_config()
    assert a is b
