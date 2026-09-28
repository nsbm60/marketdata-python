"""Watchlist + era helpers used by archive.py. Network calls faked."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from option_archive import reference
from option_archive.config import load_config
from option_archive.domain import Era

_NOW = date(2026, 9, 27)


def _cfg(tmp_path: Path, start: str = "2026-09-20"):
    body = tmp_path / "oa.yaml"
    body.write_text(
        f"""
universe: {{top_n: 100, ranking_years: [2026], index_products: [SPY], watchlist_always_include: true}}
bands:
  perishable: {{moneyness_band: 0.50, max_dte_days: 365}}
  routine: {{moneyness_band: 0.30, max_dte_days: 90}}
quotes_band: {{moneyness_band: 0.30, max_dte_days: 90}}
backfill_start_date: "{start}"
quotes_available_from: "{start}"
excluded_dates: []
roll_off: {{assumed_retention_years: 5, alert_margin_days: 60}}
s3: {{connect_timeout_seconds: 30, read_timeout_seconds: 600, max_concurrency: 16, multipart_chunksize_mb: 8, multipart_threshold_mb: 8}}
tables: {{}}
""",
        encoding="utf-8",
    )
    return load_config(body)


def test_era_for_oldest_year_is_perishable(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    # retention 5y, now 2026-09-27 -> perishable cutoff ~2022-09-28
    assert reference.era_for(date(2022, 6, 1), now=_NOW, cfg=cfg) is Era.PERISHABLE
    assert reference.era_for(date(2025, 1, 1), now=_NOW, cfg=cfg) is Era.ROUTINE


class _FakeCH:
    def __init__(self, rows: list[tuple[str]]) -> None:
        self._rows = rows
        self.last_sql = ""

    def query(self, sql: str, parameters: dict[str, Any] | None = None) -> Any:
        self.last_sql = sql
        return type("R", (), {"result_rows": self._rows})()


def test_watchlist_underlyings_distinct_upper() -> None:
    ch = _FakeCH([("nvda",), ("MU",)])
    assert reference.watchlist_underlyings(ch, table="trading.watchlist") == ("NVDA", "MU")
    assert "FINAL" in ch.last_sql


def test_week_anchor_is_monday() -> None:
    assert reference._week_anchor(date(2026, 9, 23)) == date(2026, 9, 21)  # Wed -> Mon
    assert reference._week_anchor(date(2026, 9, 21)) == date(2026, 9, 21)  # Mon -> Mon
