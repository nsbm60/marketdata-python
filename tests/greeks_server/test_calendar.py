"""SessionCache: sessions come from MDS get_trading_sessions, cached; no local holiday rules."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Sequence

import pytest

from greeks_server.calendar import ET, Session, SessionCache


def _sess(d: date) -> Session:
    return Session(
        d,
        datetime.combine(d, time(9, 30), tzinfo=ET).astimezone(timezone.utc),
        datetime.combine(d, time(16, 0), tzinfo=ET).astimezone(timezone.utc),
    )


class FakeMds:
    """Weekdays only (minus any holidays), as MDS would answer — MDS owns the holiday rules."""

    def __init__(self, holidays: "set[date] | None" = None) -> None:
        self.calls: list[tuple[date, date]] = []
        self.down = False
        self.holidays = holidays or set()

    def __call__(self, start: date, end: date) -> Sequence[Session]:
        self.calls.append((start, end))
        if self.down:
            raise RuntimeError("MDS timeout for get_trading_sessions")
        out, d = [], start
        while d <= end:
            if d.weekday() < 5 and d not in self.holidays:
                out.append(_sess(d))
            d += timedelta(days=1)
        return out


def _et(d: date, h: int, m: int) -> datetime:
    return datetime.combine(d, time(h, m), tzinfo=ET).astimezone(timezone.utc)


TUE = date(2026, 10, 20)


def test_startup_fetches_lookback_through_horizon() -> None:
    mds = FakeMds()
    SessionCache(mds, TUE)
    assert mds.calls == [(TUE - timedelta(days=10), TUE + timedelta(days=7))]


def test_startup_without_mds_raises() -> None:
    mds = FakeMds()
    mds.down = True
    with pytest.raises(RuntimeError):
        SessionCache(mds, TUE)


def test_session_bounds() -> None:
    cal = SessionCache(FakeMds(), TUE)
    assert cal.session_at(_et(TUE, 9, 29)) is None
    s = cal.session_at(_et(TUE, 9, 30))
    assert s is not None and s.date == TUE and s.open_ms == int(s.open_utc.timestamp() * 1000)
    assert cal.session_at(_et(TUE, 15, 59)) is not None
    assert cal.session_at(_et(TUE, 16, 0)) is None
    assert cal.session_at(_et(date(2026, 10, 24), 12, 0)) is None  # Saturday: MDS sent none


# 2026-11-09 is a Monday; the Friday before is 2026-11-06.
MON = date(2026, 11, 9)
FRI_BEFORE = date(2026, 11, 6)


def _pick(cal: SessionCache, dt: datetime) -> "tuple[date, datetime]":
    """The one rule the handler applies: the session, and the price time min(as_of, close)."""
    s = cal.session_as_of(dt)
    assert s is not None
    return s.date, min(dt, s.close_utc)


def test_session_as_of_is_latest_open_at_or_before() -> None:
    # The cache holds the prior Friday (10-day lookback), so no backward fetch is needed.
    cal = SessionCache(FakeMds(), MON)
    fri_close = _et(FRI_BEFORE, 16, 0)
    # Monday 00:30 and 08:00 ET: before Monday's open -> Friday's session, priced at Friday 16:00.
    assert _pick(cal, _et(MON, 0, 30)) == (FRI_BEFORE, fri_close)
    assert _pick(cal, _et(MON, 8, 0)) == (FRI_BEFORE, fri_close)
    # Monday 10:00 ET: in session -> Monday, priced at 10:00.
    assert _pick(cal, _et(MON, 10, 0)) == (MON, _et(MON, 10, 0))
    # Monday 17:00 ET: after the close -> Monday, priced at 16:00.
    assert _pick(cal, _et(MON, 17, 0)) == (MON, _et(MON, 16, 0))


def test_day_after_a_holiday_returns_the_last_trading_day_before_it() -> None:
    holiday = date(2026, 11, 26)        # a Thursday, say
    day_after = date(2026, 11, 27)      # Friday
    last_before = date(2026, 11, 25)    # Wednesday — Thursday was a holiday
    cal = SessionCache(FakeMds(holidays={holiday}), day_after)
    # Before Friday's open, the latest opened session skips the Thursday holiday.
    s = cal.session_as_of(_et(day_after, 8, 0))
    assert s is not None and s.date == last_before


def test_reads_cache_then_refreshes_with_two_days_left() -> None:
    mds = FakeMds()
    cal = SessionCache(mds, TUE)
    cal.session_at(_et(TUE + timedelta(days=5), 12, 0))   # 2 days left: cached, no fetch
    assert len(mds.calls) == 1
    later = TUE + timedelta(days=6)                        # 1 day left: refresh from that day
    assert cal.session_at(_et(later, 12, 0)) is not None
    assert mds.calls[-1] == (later - timedelta(days=10), later + timedelta(days=7))


def test_mds_outage_mid_session_does_not_affect_gating() -> None:
    mds = FakeMds()
    t = [0.0]
    cal = SessionCache(mds, TUE, monotonic=lambda: t[0])
    mds.down = True
    day6 = TUE + timedelta(days=6)                         # refresh due, MDS down, cache covers it
    assert cal.session_at(_et(day6, 12, 0)) is not None
    n = len(mds.calls)
    cal.session_at(_et(day6, 12, 1))                       # retried at most once a minute
    assert len(mds.calls) == n
    t[0] += 61
    cal.session_at(_et(day6, 12, 2))
    assert len(mds.calls) == n + 1


def test_uncovered_date_without_mds_raises() -> None:
    mds = FakeMds()
    cal = SessionCache(mds, TUE)
    mds.down = True
    with pytest.raises(RuntimeError, match="no calendar"):
        cal.session_at(_et(TUE + timedelta(days=9), 12, 0))


def test_bad_session_rejected() -> None:
    def bad(start: date, end: date) -> Sequence[Session]:
        s = _sess(start)
        return [Session(s.date, s.close_utc, s.open_utc)]
    with pytest.raises(ValueError):
        SessionCache(bad, TUE)
