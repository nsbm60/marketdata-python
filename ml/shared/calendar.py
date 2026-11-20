"""US equity regular-session trading calendar — the one shared implementation.

Trading days are Mon–Fri minus US market holidays (New Year's, MLK, Presidents, Memorial, Juneteenth
[from 2022], Independence, Labor, Thanksgiving, Christmas — each observed Sat→Fri / Sun→Mon — plus
Good Friday and the national-mourning special closures). Mirrors the Scala USMarketCalendar.
`greeks_server/calendar.py` and `iv_curve` build on this; there is no second calendar.

Juneteenth became a NYSE market holiday only in 2022 (first observed on Monday 2022-06-20); it is a
holiday only for years >= 2022. The data starts in 2016, so the earlier one-off closures (9/11 2001,
Sandy 2012) are deliberately not represented.

NYSE half days close at 13:00 ET (session_close_et / is_early_close): the day after Thanksgiving, and
July 3 / December 24 when Mon–Thu.
"""

from __future__ import annotations

from datetime import date, time, timedelta

_MONDAY, _THURSDAY = 0, 3  # date.weekday(): Mon=0 … Sun=6
_JUNETEENTH_FIRST_YEAR = 2022

# NYSE regular-session close, and the early (13:00 ET) close on half days. Mirrors
# USMarketCalendar.closeFor / isEarlyClose in the Scala repo.
REGULAR_CLOSE_ET = time(16, 0)
EARLY_CLOSE_ET = time(13, 0)

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


def is_early_close(d: date) -> bool:
    """A NYSE early close (regular session ends 13:00 ET): the day after Thanksgiving, and July 3 /
    December 24 when they fall Monday–Thursday (the Mon–Thu guard excludes the Fridays those dates
    take when the adjacent holiday is observed on them). Only on a trading day. Mirrors
    USMarketCalendar.isEarlyClose."""
    if not is_trading_day(d):
        return False
    mon_thu = d.weekday() <= _THURSDAY  # Mon=0 … Thu=3
    day_after_thanksgiving = d == _nth_weekday(d.year, 11, _THURSDAY, 4) + timedelta(days=1)
    july3 = d.month == 7 and d.day == 3 and mon_thu
    dec24 = d.month == 12 and d.day == 24 and mon_thu
    return day_after_thanksgiving or july3 or dec24


def session_close_et(d: date) -> time:
    """The regular-session close for `d`: 13:00 ET on an early-close day, else 16:00 ET."""
    return EARLY_CLOSE_ET if is_early_close(d) else REGULAR_CLOSE_ET


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


def trading_days_in(start_exclusive: date, end_inclusive: date) -> list[date]:
    """The trading days in (start_exclusive, end_inclusive], in ascending order — the list form of
    trading_days_between. Empty if the range is empty or reversed."""
    out: list[date] = []
    d = start_exclusive + timedelta(days=1)
    while d <= end_inclusive:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def previous_trading_day(d: date) -> date:
    """The latest trading day strictly before `d` (used for the gap = calendar days since it)."""
    p = d - timedelta(days=1)
    while not is_trading_day(p):
        p -= timedelta(days=1)
    return p


def next_trading_day(d: date) -> date:
    """The earliest trading day strictly after `d`."""
    n = d + timedelta(days=1)
    while not is_trading_day(n):
        n += timedelta(days=1)
    return n
