"""SessionCalendar: sessions come from MDS get_trading_sessions, cached; no local holiday rules."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Sequence

import pytest

from greeks_server.calendar import ET, Session, SessionCalendar


def _sess(d: date) -> Session:
    return Session(
        d,
        datetime.combine(d, time(9, 30), tzinfo=ET).astimezone(timezone.utc),
        datetime.combine(d, time(16, 0), tzinfo=ET).astimezone(timezone.utc),
    )


class FakeMds:
    """Weekdays only, as MDS would answer (holidays are MDS's business, not ours)."""

    def __init__(self) -> None:
        self.calls: list[tuple[date, date]] = []
        self.down = False

    def __call__(self, start: date, end: date) -> Sequence[Session]:
        self.calls.append((start, end))
        if self.down:
            raise RuntimeError("MDS timeout for get_trading_sessions")
        out, d = [], start
        while d <= end:
            if d.weekday() < 5:
                out.append(_sess(d))
            d += timedelta(days=1)
        return out


def _et(d: date, h: int, m: int) -> datetime:
    return datetime.combine(d, time(h, m), tzinfo=ET).astimezone(timezone.utc)


TUE = date(2026, 10, 20)


def test_startup_fetches_a_week() -> None:
    mds = FakeMds()
    SessionCalendar(mds, TUE)
    assert mds.calls == [(TUE, TUE + timedelta(days=7))]


def test_startup_without_mds_raises() -> None:
    mds = FakeMds()
    mds.down = True
    with pytest.raises(RuntimeError):
        SessionCalendar(mds, TUE)


def test_session_bounds() -> None:
    cal = SessionCalendar(FakeMds(), TUE)
    assert cal.session_at(_et(TUE, 9, 29)) is None
    s = cal.session_at(_et(TUE, 9, 30))
    assert s is not None and s.date == TUE and s.open_ms == int(s.open_utc.timestamp() * 1000)
    assert cal.session_at(_et(TUE, 15, 59)) is not None
    assert cal.session_at(_et(TUE, 16, 0)) is None
    assert cal.session_at(_et(date(2026, 10, 24), 12, 0)) is None  # Saturday: MDS sent none


def test_reads_cache_then_refreshes_with_two_days_left() -> None:
    mds = FakeMds()
    cal = SessionCalendar(mds, TUE)
    cal.session_at(_et(TUE + timedelta(days=5), 12, 0))   # 2 days left: cached, no fetch
    assert len(mds.calls) == 1
    later = TUE + timedelta(days=6)                        # 1 day left: refresh from that day
    assert cal.session_at(_et(later, 12, 0)) is not None
    assert mds.calls[-1] == (later, later + timedelta(days=7))


def test_mds_outage_mid_session_does_not_affect_gating() -> None:
    mds = FakeMds()
    t = [0.0]
    cal = SessionCalendar(mds, TUE, monotonic=lambda: t[0])
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
    cal = SessionCalendar(mds, TUE)
    mds.down = True
    with pytest.raises(RuntimeError, match="no calendar"):
        cal.session_at(_et(TUE + timedelta(days=9), 12, 0))


def test_bad_session_rejected() -> None:
    def bad(start: date, end: date) -> Sequence[Session]:
        s = _sess(start)
        return [Session(s.date, s.close_utc, s.open_utc)]
    with pytest.raises(ValueError):
        SessionCalendar(bad, TUE)
