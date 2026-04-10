"""
ml/models/breakout/atr_calculator.py

Average True Range (ATR) calculator for volatility measurement.
Also includes volume average calculator for volume ratio.
"""

from collections import deque
from typing import Optional

from ml.models.breakout.bar_aggregator import Bar


class ATRCalculator:
    """
    Calculates Average True Range using Wilder's smoothing.

    TR = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = EMA of TR with period N

    Uses Wilder's smoothing: ATR_t = ATR_{t-1} + (TR_t - ATR_{t-1}) / N
    """

    def __init__(self, period: int = 14):
        self.period = period
        self._prev_close: Optional[float] = None
        self._atr: Optional[float] = None
        self._count = 0

    def update(self, bar5m: Bar) -> Optional[float]:
        """
        Update ATR with new bar.

        Returns:
            Current ATR value (or None if not yet warm)
        """
        # Calculate True Range
        if self._prev_close is None:
            tr = bar5m.high - bar5m.low
        else:
            tr = max(
                bar5m.high - bar5m.low,
                abs(bar5m.high - self._prev_close),
                abs(bar5m.low - self._prev_close)
            )

        self._prev_close = bar5m.close
        self._count += 1

        # Update ATR using Wilder's smoothing
        if self._atr is None:
            self._atr = tr
        else:
            self._atr = self._atr + (tr - self._atr) / self.period

        return self._atr

    @property
    def value(self) -> Optional[float]:
        """Current ATR value."""
        return self._atr

    @property
    def is_warm(self) -> bool:
        """True if we have enough data for stable ATR (>= period bars)."""
        return self._count >= self.period

    def reset(self):
        """Reset the calculator (call at start of new session)."""
        self._prev_close = None
        self._atr = None
        self._count = 0


class VolumeAverageCalculator:
    """
    Computes rolling N-bar volume average for volume ratio calculations.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self._volumes: deque[int] = deque(maxlen=period)
        self._count = 0

    def update(self, volume: int) -> float:
        """
        Update with new bar volume.

        Returns:
            Current average volume
        """
        self._volumes.append(volume)
        self._count += 1
        return self.value

    @property
    def value(self) -> float:
        """Current average volume."""
        if not self._volumes:
            return 0.0
        return sum(self._volumes) / len(self._volumes)

    @property
    def is_warm(self) -> bool:
        """True if we have enough data (>= period bars)."""
        return self._count >= self.period

    def volume_ratio(self, current_volume: int) -> float:
        """
        Calculate ratio of current volume to average.

        Returns:
            Ratio (e.g., 1.5 means 50% above average)
        """
        avg = self.value
        if avg <= 0:
            return 0.0
        return current_volume / avg

    def reset(self):
        """Reset the calculator."""
        self._volumes.clear()
        self._count = 0
