"""US equity regular-session calendar for the greeks server (greeks-server-spec.md §4).

Mirrors the Scala USMarketCalendar: trading days are Mon–Fri minus US market holidays (New Year's,
MLK, Presidents, Memorial, Juneteenth, Independence, Labor, Thanksgiving, Christmas — each observed
Sat→Fri / Sun→Mon — plus Good Friday and the national-mourning special closures). The regular
session opens 09:30 ET.

SessionCache.most_recent_open gives the open of the most recent regular session that opened at or
before a given instant — the single "current quote" lower bound both the cold path and compute_greeks
gate on. It is the one source of the session open: never a naive 09:30 of the as_of day, which would
differ pre-market, on weekends, and on holidays.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_OPEN = time(9, 30)
_MONDAY, _THURSDAY = 0, 3  # date.weekday(): Mon=0 … Sun=6

# National days of mourning — full closures no recurring rule produces (from the Scala calendar).
_SPECIAL_CLOSURES = frozenset({date(2018, 12, 5), date(2025, 1, 9)})


def _observed(d: date) -> date:
    wd = d.weekday()
    if wd == 5:  # Saturday → observed Friday before
        return d - timedelta(days=1)
    if wd == 6:  # Sunday → observed Monday after
        return d + timedelta(days=1)
    return d


def _nth_weekday(y: int, month: int, weekday: int, n: int) -> date:
    first = date(y, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + (n - 1) * 7)


def _last_weekday(y: int, month: int, weekday: int) -> date:
    last = date(y, 12, 31) if month == 12 else date(y, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    # Anonymous Gregorian algorithm (Western), mirroring USMarketCalendar.easterSunday.
    a = year % 19
    b, c = year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return date(year, month, day)


class SessionCache:
    """The US equity regular-session calendar. Holiday sets are memoised per year."""

    def __init__(self) -> None:
        self._holidays: dict[int, frozenset[date]] = {}

    def most_recent_open(self, as_of_ms: int) -> Optional[int]:
        """The open (epoch ms) of the most recent regular session that opened at or before ``as_of``,
        or None if none is found within a two-week look-back (should never happen in practice; the
        caller then fails every contract no_current_quote rather than inventing an open)."""
        as_of = datetime.fromtimestamp(as_of_ms / 1000.0, tz=timezone.utc).astimezone(_ET)
        today = as_of.date()
        # Today counts only if it is a trading day whose 09:30 open is at or before as_of; otherwise
        # the most recent session is an earlier day (pre-market, weekend, or holiday).
        opened_today = self._is_trading_day(today) and as_of >= datetime.combine(today, _OPEN, tzinfo=_ET)
        d = today if opened_today else today - timedelta(days=1)
        for _ in range(15):
            if self._is_trading_day(d):
                return int(datetime.combine(d, _OPEN, tzinfo=_ET).timestamp() * 1000)
            d -= timedelta(days=1)
        return None

    def _is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in _SPECIAL_CLOSURES and d not in self._holiday_set(d.year)

    def _holiday_set(self, y: int) -> frozenset[date]:
        s = self._holidays.get(y)
        if s is None:
            s = self._compute_holidays(y)
            self._holidays[y] = s
        return s

    @staticmethod
    def _compute_holidays(y: int) -> frozenset[date]:
        # New Year's on a Saturday is deliberately NOT observed on the Friday before: _observed lands
        # it on Dec 31 of the PRIOR year, which is never matched by _is_trading_day (that date's own
        # year's set is used), so NYSE trades that Friday — matching the Scala calendar.
        return frozenset({
            _observed(date(y, 1, 1)),                     # New Year's
            _nth_weekday(y, 1, _MONDAY, 3),               # MLK
            _nth_weekday(y, 2, _MONDAY, 3),               # Presidents
            _last_weekday(y, 5, _MONDAY),                 # Memorial
            _observed(date(y, 6, 19)),                    # Juneteenth
            _observed(date(y, 7, 4)),                     # Independence
            _nth_weekday(y, 9, _MONDAY, 1),               # Labor
            _nth_weekday(y, 11, _THURSDAY, 4),            # Thanksgiving
            _observed(date(y, 12, 25)),                   # Christmas
            _easter_sunday(y) - timedelta(days=2),        # Good Friday
        })
