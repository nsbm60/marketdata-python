"""Options session calendar — a port of Scala calendar/USMarketCalendar.sessionFor.

The server gates on this and nothing else (greeks-server-spec.md §5.3): one check on every loop
iteration covers a pre-open start, a mid-session start and a restart. It mirrors the Scala
calendar exactly, including what that calendar leaves out:

  - Trading days: Mon–Fri minus New Year's, MLK, Presidents, Good Friday, Memorial, Juneteenth,
    Independence, Labor, Thanksgiving, Christmas (Sat → Fri, Sun → Mon observed). A Saturday
    Jan 1 is not observed on the Friday before (NYSE: 2021-12-31 traded); the Scala calendar had
    that wrong and was fixed with this port.
  - Session 09:30–16:00 America/New_York. No early closes.

A change to the Scala calendar must be made here too; tests/greeks_server/test_calendar.py pins
the 2026 and 2027 holiday lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
OPEN_ET = time(9, 30)
CLOSE_ET = time(16, 0)

_MON, _THU, _SAT, _SUN = 0, 3, 5, 6


@dataclass(frozen=True)
class Session:
    """One regular options session. open_utc <= t < close_utc is "in session"."""

    date: date
    open_utc: datetime
    close_utc: datetime

    @property
    def open_ms(self) -> int:
        return int(self.open_utc.timestamp() * 1000)

    def contains(self, t_utc: datetime) -> bool:
        return self.open_utc <= t_utc < self.close_utc


def _observed(d: date) -> date:
    if d.weekday() == _SAT:
        return d - timedelta(days=1)
    if d.weekday() == _SUN:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    add = (weekday - first.weekday()) % 7
    return first + timedelta(days=add + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm (same as the Scala calendar)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = ((h + ll - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@lru_cache(maxsize=8)
def holidays(year: int) -> frozenset[date]:
    """The Scala calendar's holiday set for one year."""
    days = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, _MON, 3),
        _nth_weekday(year, 2, _MON, 3),
        _last_weekday(year, 5, _MON),
        _observed(date(year, 6, 19)),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, _MON, 1),
        _nth_weekday(year, 11, _THU, 4),
        _observed(date(year, 12, 25)),
        _easter_sunday(year) - timedelta(days=2),
    }
    return frozenset(days)


def is_trading_day(d: date) -> bool:
    return d.weekday() < _SAT and d not in holidays(d.year)


def session_for(d: date) -> Optional[Session]:
    if not is_trading_day(d):
        return None
    return Session(
        date=d,
        open_utc=datetime.combine(d, OPEN_ET, tzinfo=ET).astimezone(timezone.utc),
        close_utc=datetime.combine(d, CLOSE_ET, tzinfo=ET).astimezone(timezone.utc),
    )


def session_at(t_utc: datetime) -> Optional[Session]:
    """The session in progress at ``t_utc``, or None outside regular hours."""
    if t_utc.tzinfo is None:
        raise ValueError("t_utc must be timezone-aware")
    s = session_for(t_utc.astimezone(ET).date())
    if s is not None and s.contains(t_utc):
        return s
    return None
