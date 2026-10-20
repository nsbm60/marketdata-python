"""Options session calendar, from MDS (greeks-server-spec.md §5.3).

The calendar is Scala's alone: Python never re-implements holiday or session rules. Sessions come
from MDS's get_trading_sessions op (it wraps USMarketCalendar.sessionFor) through the existing
client, ml.shared.mds_client.get_trading_sessions.

Cache: fetched at startup for today through today + 7 days, and again when fewer than two days
remain. session_at() reads the cache, so an MDS outage mid-session doesn't affect gating. A
failed refresh is logged and retried at most once a minute while the cache still covers today;
a date the cache doesn't cover and can't fetch raises (the server exits and restarts waiting for
MDS — it doesn't run without a calendar).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Optional, Sequence
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
HORIZON_DAYS = 7
REFRESH_WHEN_DAYS_LEFT = 2
RETRY_S = 60.0


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


Fetch = Callable[[date, date], Sequence[Session]]


class SessionCache:
    """Cached MDS sessions. Thread-safe (compute thread and router thread both ask)."""

    def __init__(self, fetch: Fetch, today: date, *, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._fetch = fetch
        self._mono = monotonic
        self._lock = threading.Lock()
        self._sessions: dict[date, Session] = {}
        self._covered_from = today
        self._covered_to = today - timedelta(days=1)  # empty until the first load
        self._last_try = float("-inf")
        self._load(today)  # startup: raises if MDS can't answer

    def _load(self, start: date) -> None:
        end = start + timedelta(days=HORIZON_DAYS)
        sessions = self._fetch(start, end)
        for s in sessions:
            if not start <= s.date <= end or s.open_utc.tzinfo is None or not s.open_utc < s.close_utc:
                raise ValueError(f"get_trading_sessions returned a bad session: {s}")
        self._sessions = {s.date: s for s in sessions}
        self._covered_from, self._covered_to = start, end
        log.info("calendar: %d sessions %s..%s from MDS", len(sessions), start, end)

    def session_at(self, t_utc: datetime) -> Optional[Session]:
        """The session in progress at ``t_utc``, or None outside regular hours."""
        if t_utc.tzinfo is None:
            raise ValueError("t_utc must be timezone-aware")
        d = t_utc.astimezone(ET).date()
        with self._lock:
            if (self._covered_to - d).days < REFRESH_WHEN_DAYS_LEFT:
                self._refresh(d)
            s = self._sessions.get(d)
        if s is not None and s.contains(t_utc):
            return s
        return None

    def _refresh(self, d: date) -> None:
        covered = self._covered_from <= d <= self._covered_to
        now = self._mono()
        if covered and now - self._last_try < RETRY_S:
            return
        self._last_try = now
        try:
            self._load(d)
        except Exception as e:
            if not covered:
                raise RuntimeError(f"no calendar for {d}: MDS get_trading_sessions failed: {e}") from e
            log.warning("calendar refresh failed (cache still covers %s): %s", d, e)


def mds_fetch(router: str) -> Fetch:
    """Fetch through the existing MDS client (ml.shared.mds_client.get_trading_sessions)."""
    from ml.shared.mds_client import get_trading_sessions

    def fetch(start: date, end: date) -> Sequence[Session]:
        return [Session(s.date, s.open_utc, s.close_utc) for s in get_trading_sessions(router, start, end)]

    return fetch
