"""Session calendar: a port of Scala USMarketCalendar.sessionFor (holidays, 09:30–16:00 ET)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from greeks_server.calendar import holidays, is_trading_day, session_at, session_for

ET = ZoneInfo("America/New_York")

# NYSE full-day closures (published schedule), which the Scala calendar reproduces.
NYSE_2026 = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
}
NYSE_2027 = {
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31),
    date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24),
}


def test_holidays_2026() -> None:
    assert holidays(2026) == NYSE_2026


def test_holidays_2027() -> None:
    # Jan 1 2028 is a Saturday: the Scala rule adds Fri Dec 31 2027 (NYSE itself stays open).
    assert holidays(2027) == NYSE_2027 | {date(2027, 12, 31)}


def test_weekends_and_holidays_have_no_session() -> None:
    assert session_for(date(2026, 10, 24)) is None  # Saturday
    assert session_for(date(2026, 10, 25)) is None  # Sunday
    assert session_for(date(2026, 11, 26)) is None  # Thanksgiving
    assert is_trading_day(date(2026, 11, 27))       # day after: open (no early close modelled)


def _et(y: int, mo: int, d: int, h: int, mi: int, s: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, s, tzinfo=ET).astimezone(timezone.utc)


def test_session_bounds_edt() -> None:
    assert session_at(_et(2026, 10, 20, 9, 29, 59)) is None
    s = session_at(_et(2026, 10, 20, 9, 30))
    assert s is not None and s.date == date(2026, 10, 20)
    assert s.open_utc == datetime(2026, 10, 20, 13, 30, tzinfo=timezone.utc)
    assert session_at(_et(2026, 10, 20, 15, 59, 59)) is not None
    assert session_at(_et(2026, 10, 20, 16, 0)) is None


def test_session_bounds_est() -> None:
    s = session_at(_et(2026, 12, 1, 12, 0))
    assert s is not None
    assert s.open_utc == datetime(2026, 12, 1, 14, 30, tzinfo=timezone.utc)
    assert s.close_utc == datetime(2026, 12, 1, 21, 0, tzinfo=timezone.utc)
    assert s.open_ms == int(s.open_utc.timestamp() * 1000)


def test_every_day_of_2026_matches_weekday_minus_holidays() -> None:
    d = date(2026, 1, 1)
    while d.year == 2026:
        assert is_trading_day(d) == (d.weekday() < 5 and d not in NYSE_2026)
        d += timedelta(days=1)
