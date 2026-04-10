"""
ml/models/breakout/bar_aggregator.py

Aggregates 1-minute bars into 5-minute bars on proper market boundaries.
All timestamps are in ET (America/New_York).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


class Bar1m(Protocol):
    """Protocol for 1-minute bar input."""
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float


@dataclass(slots=True)
class Bar:
    """5-minute OHLCV bar aggregated from 1-minute bars."""
    ts: datetime  # Start of the 5-minute period in ET
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float


class BarAggregator:
    """
    Aggregates 1-minute bars into 5-minute bars.

    Boundaries are at 5-minute intervals: 9:30, 9:35, 9:40, ...
    A 5-minute bar is completed when we receive a bar from the next period.

    All timestamps must be in ET (America/New_York).
    """

    def __init__(self, symbol: str, timeframe_minutes: int = 5):
        self.symbol = symbol
        self.timeframe = timeframe_minutes
        self._current_period: Optional[datetime] = None
        self._pending_bars: list[Bar1m] = []
        self._completed_bars: list[Bar] = []

    def add_bar(self, bar: Bar1m) -> Optional[Bar]:
        """
        Add a 1-minute bar. Returns completed 5m bar if period boundary crossed.

        Args:
            bar: 1-minute bar with timestamp in ET

        Returns:
            Completed 5m bar if a period just ended, None otherwise
        """
        period_start = self._get_period_start(bar.ts)

        # First bar or same period
        if self._current_period is None:
            self._current_period = period_start
            self._pending_bars = [bar]
            return None

        # Same period - accumulate
        if period_start == self._current_period:
            self._pending_bars.append(bar)
            return None

        # New period - finalize previous and start new
        completed = self._finalize_period()
        self._current_period = period_start
        self._pending_bars = [bar]
        return completed

    def get_bars(self, n: Optional[int] = None) -> list[Bar]:
        """Return list of completed 5-minute bars (oldest first)."""
        if n is None:
            return list(self._completed_bars)
        return self._completed_bars[-n:] if n > 0 else []

    def get_last_bar(self) -> Optional[Bar]:
        """Return the most recently completed 5-minute bar."""
        return self._completed_bars[-1] if self._completed_bars else None

    def bar_count(self) -> int:
        """Return number of completed 5-minute bars."""
        return len(self._completed_bars)

    def clear(self):
        """Clear all state (call at start of new session)."""
        self._current_period = None
        self._pending_bars = []
        self._completed_bars = []

    def _get_period_start(self, ts: datetime) -> datetime:
        """
        Round timestamp down to the start of its 5-minute period.

        Input must be in ET. Returns ET timestamp.
        Example: 9:32 ET -> 9:30 ET, 9:37 ET -> 9:35 ET
        """
        minute = ts.minute - (ts.minute % self.timeframe)
        return ts.replace(minute=minute, second=0, microsecond=0)

    def _finalize_period(self) -> Optional[Bar]:
        """Convert pending 1-minute bars into a completed 5-minute bar."""
        if not self._pending_bars:
            return None

        bars = self._pending_bars

        # OHLCV aggregation
        open_price = bars[0].open
        high_price = max(b.high for b in bars)
        low_price = min(b.low for b in bars)
        close_price = bars[-1].close
        total_volume = sum(b.volume for b in bars)

        # Volume-weighted average price
        if total_volume > 0:
            vwap = sum(b.vwap * b.volume for b in bars) / total_volume
        else:
            vwap = close_price

        bar = Bar(
            ts=self._current_period,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=total_volume,
            vwap=vwap,
        )

        self._completed_bars.append(bar)
        return bar
