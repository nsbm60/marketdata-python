"""ACT/365 second-resolution T tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from greeks.solver.time import expiry_instant_utc, years_from_seconds, years_to_expiry

_ET = ZoneInfo("America/New_York")


def test_expiry_instant_is_16_et() -> None:
    exp = expiry_instant_utc(date(2026, 6, 5))
    local = exp.astimezone(_ET)
    assert local.hour == 16
    assert local.minute == 0
    assert local.date() == date(2026, 6, 5)


def test_exact_one_calendar_day() -> None:
    # From 16:00 ET day D-1 to 16:00 ET day D = 86400 seconds = 1/365 years
    expiry = date(2026, 6, 5)
    as_of = expiry_instant_utc(expiry) - timedelta(days=1)
    t = years_to_expiry(as_of, expiry)
    assert t == pytest.approx(1.0 / 365.0, rel=0, abs=1e-15)


def test_sub_day_points() -> None:
    expiry = date(2026, 6, 5)
    exp_utc = expiry_instant_utc(expiry)
    for minutes in (15, 30, 60, 120, 240, 390):  # 6.5h = 390m
        as_of = exp_utc - timedelta(minutes=minutes)
        t = years_to_expiry(as_of, expiry)
        expected = (minutes * 60.0) / (365.0 * 86400.0)
        assert t == pytest.approx(expected, rel=0, abs=1e-18)


def test_past_expiry_is_zero() -> None:
    expiry = date(2026, 6, 5)
    as_of = expiry_instant_utc(expiry) + timedelta(seconds=1)
    assert years_to_expiry(as_of, expiry) == 0.0


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        years_to_expiry(datetime(2026, 6, 1, 12, 0, 0), date(2026, 6, 5))


def test_years_from_seconds() -> None:
    assert years_from_seconds(365 * 86400) == pytest.approx(1.0)
