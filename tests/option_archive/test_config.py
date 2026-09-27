"""Config chokepoint behaviour for option_archive (PR0)."""

from __future__ import annotations

from datetime import time
from pathlib import Path

import pytest

from option_archive.config import load_config
from option_archive.domain import Era, ScheduleWindowKind

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
quotes_band: {moneyness_band: 0.30, max_dte_days: 90}
backfill_start_date: "2022-03-07"
quotes_available_from: "2022-03-07"
excluded_dates: ["2026-06-08"]
roll_off: {assumed_retention_years: 5, alert_margin_days: 60}
schedule_windows:
  - {kind: aggressive, start_et: "20:00", end_et: "04:00", requests_per_sec: 5.0, worker_count: 4}
  - {kind: polite, start_et: "04:00", end_et: "20:00", requests_per_sec: 1.0, worker_count: 1}
queue_db_path: "{queue_db}"
tables:
  option_trades: "trading.option_trades"
  splits: "trading.splits"
  universe_ranking: "trading.option_universe_ranking"
  dividend: "trading.dividend"
  option_contract: "trading.option_contract"
"""


def _write(tmp_path: Path, body: str, queue_db: str | None = None) -> Path:
    qdb = queue_db if queue_db is not None else str(tmp_path / "queue.db")
    p = tmp_path / "option_archive.yaml"
    p.write_text(body.replace("{queue_db}", qdb), encoding="utf-8")
    return p


def test_default_shipped_config_loads() -> None:
    # The committed config/option_archive.yaml must always be loadable.
    cfg = load_config()
    assert cfg.universe.top_n == 100
    assert cfg.band_for(Era.PERISHABLE).moneyness_band == 0.50
    assert cfg.band_for(Era.ROUTINE).max_dte_days == 90
    assert cfg.quotes_band.moneyness_band == 0.30
    assert cfg.tables.dividend == "trading.dividend"
    assert cfg.queue_db_path.is_absolute()


def test_temp_config_parses_all_fields(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _BASE))
    assert cfg.universe.ranking_years == (2022, 2023)
    assert cfg.universe.index_products == ("SPY", "QQQ")
    assert len(cfg.schedule_windows) == 2
    # window kind is an enum and times are real time objects, not strings
    assert cfg.schedule_windows[0].kind is ScheduleWindowKind.AGGRESSIVE
    assert cfg.schedule_windows[0].start_et == time(20, 0)
    assert cfg.schedule_windows[0].end_et == time(4, 0)
    assert cfg.roll_off.alert_margin_days == 60
    assert cfg.excluded_dates[0].isoformat() == "2026-06-08"
    assert cfg.backfill_start_date.isoformat() == "2022-03-07"
    assert cfg.quotes_available_from.isoformat() == "2022-03-07"


def test_queue_db_inside_repo_is_rejected(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    inside = str(repo_root / "data" / "queue.db")
    with pytest.raises(ValueError, match="OUTSIDE the repo"):
        load_config(_write(tmp_path, _BASE, queue_db=inside))


def test_relative_queue_db_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        load_config(_write(tmp_path, _BASE, queue_db="data/queue.db"))


def test_missing_era_band_is_rejected(tmp_path: Path) -> None:
    body = _BASE.replace(
        "  routine: {moneyness_band: 0.30, max_dte_days: 90}\n", ""
    )
    with pytest.raises(ValueError, match="bands.routine"):
        load_config(_write(tmp_path, body))
