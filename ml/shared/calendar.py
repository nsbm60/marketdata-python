"""US equity regular-session trading calendar — the one shared implementation.

Trading days are Mon–Fri minus US market holidays (New Year's, MLK, Presidents, Memorial, Juneteenth
[from 2022], Independence, Labor, Thanksgiving, Christmas — each observed Sat→Fri / Sun→Mon — plus
Good Friday and the national-mourning special closures). Mirrors the Scala USMarketCalendar.
`greeks_server/calendar.py` and `iv_curve` build on this; there is no second calendar.

Juneteenth became a NYSE market holiday only in 2022 (first observed on Monday 2022-06-20); it is a
holiday only for years >= 2022. The data starts in 2016, so the earlier one-off closures (9/11 2001,
Sandy 2012) are deliberately not represented.
"""

from __future__ import annotations

from datetime import date, timedelta

_MONDAY, _THURSDAY = 0, 3  # date.weekday(): Mon=0 … Sun=6
_JUNETEENTH_FIRST_YEAR = 2022

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


_holidays: dict[int, frozenset[date]] = {}


def _holiday_set(y: int) -> frozenset[date]:
    s = _holidays.get(y)
    if s is None:
        s = _compute_holidays(y)
        _holidays[y] = s
    return s


def _compute_holidays(y: int) -> frozenset[date]:
    # New Year's on a Saturday is deliberately NOT observed on the Friday before: _observed lands it
    # on Dec 31 of the PRIOR year, which is never matched by is_trading_day (that date's own year's
    # set is used), so NYSE trades that Friday — matching the Scala calendar.
    days = [
        _observed(date(y, 1, 1)),             # New Year's
        _nth_weekday(y, 1, _MONDAY, 3),       # MLK
        _nth_weekday(y, 2, _MONDAY, 3),       # Presidents
        _last_weekday(y, 5, _MONDAY),         # Memorial
        _observed(date(y, 7, 4)),             # Independence
        _nth_weekday(y, 9, _MONDAY, 1),       # Labor
        _nth_weekday(y, 11, _THURSDAY, 4),    # Thanksgiving
        _observed(date(y, 12, 25)),           # Christmas
        _easter_sunday(y) - timedelta(days=2),  # Good Friday
    ]
    # Juneteenth: a NYSE holiday only from 2022 (first observed Monday 2022-06-20).
    if y >= _JUNETEENTH_FIRST_YEAR:
        days.append(_observed(date(y, 6, 19)))
    return frozenset(days)


def is_trading_day(d: date) -> bool:
    """A US equity regular session trades on `d`: a weekday that is neither a special closure nor a
    recurring/observed market holiday for its year."""
    return d.weekday() < 5 and d not in _SPECIAL_CLOSURES and d not in _holiday_set(d.year)


def trading_days_between(start_exclusive: date, end_inclusive: date) -> int:
    """The number of trading days in (start_exclusive, end_inclusive] — i.e. trading days strictly
    after `start_exclusive`, up to and including `end_inclusive`. 0 if the range is empty."""
    n = 0
    d = start_exclusive + timedelta(days=1)
    while d <= end_inclusive:
        if is_trading_day(d):
            n += 1
        d += timedelta(days=1)
    return n
