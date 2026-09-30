"""Config chokepoint behaviour for option_archive."""

from __future__ import annotations

from pathlib import Path

import pytest

from option_archive.config import load_config
from option_archive.domain import Era

# A valid config body reused across tests; individual tests mutate a copy.
_BASE = """
universe:
  top_n: 100
  ranking_years: [2022, 2023]
  index_products: [SPY, QQQ]
  watchlist_always_include: true
bands:
  perishable: {moneyness_band: 0.50, max_dte_days: 365}
  routine: {moneyness_band: 0.30, max_dte_days: 90}
band_overrides:
  SPY: {moneyness_pct: 10, dte_max: 90}
expiry_weekday_exclude:
  SPY: [Tue, Thu]
quotes_band: {moneyness_band: 0.30, max_dte_days: 90}
backfill_start_date: "2022-03-07"
quotes_available_from: "2022-03-07"
excluded_dates: ["2026-06-08"]
roll_off: {assumed_retention_years: 5, alert_margin_days: 60}
s3: {connect_timeout_seconds: 30, read_timeout_seconds: 600, max_concurrency: 16, multipart_chunksize_mb: 8, multipart_threshold_mb: 8}
tables:
  option_trade: "trading.option_trade"
  split: "trading.split"
  universe_ranking: "trading.option_universe_ranking"
  dividend: "trading.dividend"
  option_contract: "trading.option_contract"
  option_contract_asof: "trading.option_contract_asof"
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "option_archive.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_default_shipped_config_loads() -> None:
    # The committed config/option_archive.yaml must always be loadable.
    cfg = load_config()
    assert cfg.universe.top_n == 100
    assert cfg.band_for(Era.PERISHABLE).moneyness_band == 0.50
    assert cfg.band_for(Era.ROUTINE).max_dte_days == 90
    assert cfg.quotes_band.moneyness_band == 0.30
    assert cfg.tables.dividend == "trading.dividend"
    assert cfg.tables.option_contract_asof == "trading.option_contract_asof"
    assert cfg.s3.max_concurrency == 16
    assert cfg.quote_pool_size == 16


def test_temp_config_parses_all_fields(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _BASE))
    assert cfg.universe.ranking_years == (2022, 2023)
    assert cfg.universe.index_products == ("SPY", "QQQ")
    assert cfg.roll_off.alert_margin_days == 60
    assert cfg.s3.read_timeout == 600.0
    assert cfg.excluded_dates[0].isoformat() == "2026-06-08"
    assert cfg.backfill_start_date.isoformat() == "2022-03-07"
    assert cfg.quotes_available_from.isoformat() == "2022-03-07"


def test_band_override_applies_in_both_eras(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _BASE))
    # SPY overridden to ±10%/90 regardless of era; NVDA falls through to the per-era band
    spy_p = cfg.band_for_underlying("SPY", Era.PERISHABLE)
    spy_r = cfg.band_for_underlying("SPY", Era.ROUTINE)
    assert spy_p == spy_r and spy_p.moneyness_band == 0.10 and spy_p.max_dte_days == 90
    assert cfg.band_for_underlying("NVDA", Era.PERISHABLE) == cfg.band_for(Era.PERISHABLE)


def test_expiry_weekday_exclude(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _BASE))
    assert cfg.excluded_expiry_weekdays("SPY") == frozenset({1, 3})  # Tue, Thu
    assert cfg.excluded_expiry_weekdays("NVDA") == frozenset()       # no exclusion


def test_missing_era_band_is_rejected(tmp_path: Path) -> None:
    body = _BASE.replace(
        "  routine: {moneyness_band: 0.30, max_dte_days: 90}\n", ""
    )
    with pytest.raises(ValueError, match="bands.routine"):
        load_config(_write(tmp_path, body))
