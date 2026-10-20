"""Shared test fixtures for greeks_server tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from greeks.domain import Dividend
from greeks_server.compute import Methodology
from greeks_server.inputs import SessionInputs

SESSION = date(2026, 10, 20)
# 2026-10-20 11:00 ET (EDT) = 15:00 UTC
NOW = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)
NOW_MS = int(NOW.timestamp() * 1000)
OPEN_MS = int(datetime(2026, 10, 20, 13, 30, tzinfo=timezone.utc).timestamp() * 1000)

M = Methodology(methodology_version="v2", t_floor_minutes=15.0, expiry_time_et="16:00")


def sofr_flat(rate: float = 0.043, days: int = 30) -> dict[date, float]:
    return {SESSION - timedelta(days=i): rate for i in range(1, days)}


def inputs(dividends: dict[str, tuple[Dividend, ...]] | None = None) -> SessionInputs:
    return SessionInputs(session_date=SESSION, sofr=sofr_flat(), dividends=dividends or {})
