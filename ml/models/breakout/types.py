"""
ml/models/breakout/types.py

Shared types for the breakout detector. Extracted from bar_aggregator.py,
ema_ribbon.py, and atr_calculator.py during the MDS indicator refactor.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


@dataclass(slots=True)
class Bar:
    """OHLCV bar (any timeframe)."""
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float


@dataclass(slots=True)
class PivotPoint:
    """Confirmed pivot point published by MDS."""
    direction: str          # "high" or "low"
    price: float
    pivot_ts: datetime      # timestamp of the pivot bar (UTC)
    confirmed_ts: datetime  # timestamp when confirmed, N bars later (UTC)
    pivot_bar_index: int


class RibbonState(Enum):
    """EMA ribbon alignment state."""
    ORDERED_BULLISH = "ordered_bullish"   # EMA10 > EMA15 > EMA20 > EMA25 > EMA30
    ORDERED_BEARISH = "ordered_bearish"   # EMA10 < EMA15 < EMA20 < EMA25 < EMA30
    TRANSITIONAL = "transitional"         # Neither fully ordered


class VolumeAverageCalculator:
    """Computes rolling N-bar volume average for volume ratio calculations."""

    def __init__(self, period: int = 20):
        self.period = period
        self._volumes: deque[int] = deque(maxlen=period)
        self._count = 0

    def update(self, volume: int) -> float:
        """Update with new bar volume. Returns current average."""
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
        """Ratio of current volume to average."""
        avg = self.value
        if avg <= 0:
            return 0.0
        return current_volume / avg

    def reset(self):
        """Reset the calculator."""
        self._volumes.clear()
        self._count = 0
