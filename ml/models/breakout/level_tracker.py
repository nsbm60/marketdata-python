"""
ml/models/breakout/level_tracker.py

Tracks session high/low levels and their age for breakout detection.
"""

from datetime import datetime
from typing import Optional

from ml.models.breakout.bar_aggregator import Bar


class LevelTracker:
    """
    Tracks session high and low with time-since-established.

    For breakout detection, we want to know:
    - Current session high/low
    - How long ago the high/low was set (level age)
    - Prior session high/low for context
    """

    def __init__(
        self,
        symbol: str,
        timeframe_minutes: int = 5,
        prior_close: Optional[float] = None,
        prior_session_high: Optional[float] = None,
        prior_session_low: Optional[float] = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe_minutes

        # Prior session data
        self.prior_close = prior_close
        self.prior_session_high = prior_session_high
        self.prior_session_low = prior_session_low

        # Current session tracking
        self._high_price: Optional[float] = None
        self._high_ts: Optional[datetime] = None
        self._low_price: Optional[float] = None
        self._low_ts: Optional[datetime] = None
        self._last_bar_ts: Optional[datetime] = None
        self._bar_count: int = 0

    def update(self, bar5m: Bar) -> None:
        """
        Update levels with new 5-minute bar.

        Args:
            bar5m: Completed 5-minute bar
        """
        self._bar_count += 1
        self._last_bar_ts = bar5m.ts

        # Update high
        if self._high_price is None or bar5m.high > self._high_price:
            self._high_price = bar5m.high
            self._high_ts = bar5m.ts

        # Update low
        if self._low_price is None or bar5m.low < self._low_price:
            self._low_price = bar5m.low
            self._low_ts = bar5m.ts

    @property
    def high_price(self) -> Optional[float]:
        """Current session high."""
        return self._high_price

    @property
    def low_price(self) -> Optional[float]:
        """Current session low."""
        return self._low_price

    def high_age_minutes(self) -> int:
        """Minutes since session high was established."""
        if self._high_ts is None or self._last_bar_ts is None:
            return 0
        delta = self._last_bar_ts - self._high_ts
        return int(delta.total_seconds() / 60)

    def low_age_minutes(self) -> int:
        """Minutes since session low was established."""
        if self._low_ts is None or self._last_bar_ts is None:
            return 0
        delta = self._last_bar_ts - self._low_ts
        return int(delta.total_seconds() / 60)

    def high_age_bars(self) -> int:
        """Bars since session high was established."""
        return self.high_age_minutes() // self.timeframe

    def low_age_bars(self) -> int:
        """Bars since session low was established."""
        return self.low_age_minutes() // self.timeframe

    def is_new_high(self, bar5m: Bar) -> bool:
        """Check if bar made a new session high."""
        if self._high_price is None:
            return True
        return bar5m.high > self._high_price

    def is_new_low(self, bar5m: Bar) -> bool:
        """Check if bar made a new session low."""
        if self._low_price is None:
            return True
        return bar5m.low < self._low_price

    def clear(self) -> None:
        """Clear session levels (call at start of new session)."""
        self._high_price = None
        self._high_ts = None
        self._low_price = None
        self._low_ts = None
        self._last_bar_ts = None
        self._bar_count = 0

    def bar_count(self) -> int:
        """Number of bars processed."""
        return self._bar_count
