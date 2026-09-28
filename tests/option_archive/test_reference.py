"""Enumeration/seeding behaviour (PR2). Network calls are faked."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from greeks.domain import OptionRight
from greeks.pull.contracts import ContractRef
from option_archive import reference
from option_archive.config import load_config
from option_archive.domain import Era, TaskStatus
from option_archive.queue import WorkQueue

_NOW = date(2026, 9, 27)


def _cfg(tmp_path: Path, start: str = "2026-09-20"):
    body = (tmp_path / "oa.yaml")
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
queue: {{lease_seconds: 1800, max_attempts: 5, backoff_base_seconds: 60}}
s3: {{connect_timeout_seconds: 30, read_timeout_seconds: 600, max_concurrency: 16, multipart_chunksize_mb: 8, multipart_threshold_mb: 8}}
schedule:
  - {{kind: polite, start_et: "07:00", requests_per_sec: 1.0, worker_count: 1}}
  - {{kind: aggressive, start_et: "20:00", requests_per_sec: 5.0, worker_count: 4}}
queue_db_path: "{tmp_path / 'q.db'}"
tables: {{}}
""",
        encoding="utf-8",
    )
    return load_config(body)


def _queue(tmp_path: Path) -> WorkQueue:
    return WorkQueue(
        tmp_path / "q.db",
        lease=timedelta(minutes=10),
        max_attempts=3,
        backoff_base=timedelta(seconds=60),
    )


def _contract(osi: str, expiry: date, strike: float, shares: int = 100) -> ContractRef:
    return ContractRef(
        osi=osi,
        massive_ticker="O:" + osi,
        underlying="NVDA",
        expiry=expiry,
        strike=strike,
        right=OptionRight.CALL,
        shares_per_contract=shares,
        exercise_style="american",
        primary_exchange=None,
    )


# -- era_for ------------------------------------------------------------------


def test_era_for_oldest_year_is_perishable(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    # retention 5y, now 2026-09-27 -> perishable cutoff ~2022-09-28
    assert era_for_result(cfg, date(2022, 6, 1)) is Era.PERISHABLE
    assert era_for_result(cfg, date(2025, 1, 1)) is Era.ROUTINE


def era_for_result(cfg: Any, d: date) -> Era:
    return reference.era_for(d, now=_NOW, cfg=cfg)


# -- watchlist_underlyings ----------------------------------------------------


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


# -- seed_watchlist (fakes for Alpaca + Massive) ------------------------------


class _Bar:
    def __init__(self, d: date, close: float) -> None:
        self.timestamp = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        self.close = close


class _FakeAlpaca:
    def __init__(self, closes: dict[date, float]) -> None:
        self._closes = closes

    def get_stock_bars(self, req: Any) -> Any:
        bars = [_Bar(d, c) for d, c in sorted(self._closes.items())]
        return type("B", (), {"data": {"NVDA": bars}})()


def test_seed_enqueues_in_band_and_excludes_nonstandard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, start="2026-09-21")
    q = _queue(tmp_path)

    # one trading day with spot 100; NVDA weekday 2026-09-21 is a Monday
    closes = {date(2026, 9, 21): 100.0}
    monkeypatch.setattr(reference, "watchlist_underlyings", lambda *a, **k: ("NVDA",))
    monkeypatch.setattr(reference, "_fetch_daily_raw_closes", lambda *a, **k: closes)

    contracts = [
        _contract("NVDA261016C00100000", date(2026, 10, 16), 100.0),  # ATM, standard -> eligible
        _contract("NVDA261016C00300000", date(2026, 10, 16), 300.0),  # +200% -> out of band
        _contract("NVDA261016C00100000", date(2026, 10, 16), 100.0, shares=10),  # nonstandard
    ]
    monkeypatch.setattr(reference, "_contracts_asof", lambda *a, **k: contracts)

    rep = reference.seed_watchlist(
        q, _FakeCH([]), cfg, massive_api_key="k", alpaca=_FakeAlpaca(closes), now=_NOW
    )
    assert rep.trading_days == 1
    assert rep.eligible_contract_days == 1  # only the ATM standard contract
    assert rep.nonstandard_excluded == 1
    assert rep.enqueued == 1
    assert q.counts()[TaskStatus.PENDING] == 1


def test_seed_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, start="2026-09-21")
    q = _queue(tmp_path)
    closes = {date(2026, 9, 21): 100.0}
    monkeypatch.setattr(reference, "watchlist_underlyings", lambda *a, **k: ("NVDA",))
    monkeypatch.setattr(reference, "_fetch_daily_raw_closes", lambda *a, **k: closes)
    monkeypatch.setattr(
        reference,
        "_contracts_asof",
        lambda *a, **k: [_contract("NVDA261016C00100000", date(2026, 10, 16), 100.0)],
    )
    kw = dict(massive_api_key="k", alpaca=_FakeAlpaca(closes), now=_NOW)
    first = reference.seed_watchlist(q, _FakeCH([]), cfg, **kw)  # type: ignore[arg-type]
    second = reference.seed_watchlist(q, _FakeCH([]), cfg, **kw)  # type: ignore[arg-type]
    assert first.enqueued == 1
    assert second.enqueued == 0  # already present — INSERT OR IGNORE
    assert q.counts()[TaskStatus.PENDING] == 1
