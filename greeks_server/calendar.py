"""US equity regular-session open lookup for the greeks server (greeks-server-spec.md §4).

The trading-day/holiday rules are the one shared implementation in `ml.shared.calendar`; this module
adds only the regular-session open (09:30 ET) and the most-recent-open lookup.

SessionCache.most_recent_open gives the open of the most recent regular session that opened at or
before a given instant — the single "current quote" lower bound both the cold path and compute_greeks
gate on. It is the one source of the session open: never a naive 09:30 of the as_of day, which would
differ pre-market, on weekends, and on holidays.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from ml.shared.calendar import is_trading_day

_ET = ZoneInfo("America/New_York")
_OPEN = time(9, 30)


class SessionCache:
    """The US equity regular-session open, over the one shared trading-day calendar."""

    def most_recent_open(self, as_of_ms: int) -> Optional[int]:
        """The open (epoch ms) of the most recent regular session that opened at or before ``as_of``,
        or None if none is found within a two-week look-back (should never happen in practice; the
        caller then fails every contract no_current_quote rather than inventing an open)."""
        as_of = datetime.fromtimestamp(as_of_ms / 1000.0, tz=timezone.utc).astimezone(_ET)
        today = as_of.date()
        # Today counts only if it is a trading day whose 09:30 open is at or before as_of; otherwise
        # the most recent session is an earlier day (pre-market, weekend, or holiday).
        opened_today = is_trading_day(today) and as_of >= datetime.combine(today, _OPEN, tzinfo=_ET)
        d = today if opened_today else today - timedelta(days=1)
        for _ in range(15):
            if is_trading_day(d):
                return int(datetime.combine(d, _OPEN, tzinfo=_ET).timestamp() * 1000)
            d -= timedelta(days=1)
        return None
