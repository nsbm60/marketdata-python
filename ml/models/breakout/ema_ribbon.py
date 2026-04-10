"""
ml/models/breakout/ema_ribbon.py

EMA ribbon indicator for breakout detection.
Uses 5 EMAs on 5-minute bar closes to determine trend state.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ml.models.breakout.bar_aggregator import Bar5m

EMA_PERIODS = [10, 15, 20, 25, 30]


class RibbonState(Enum):
    """Current state of the EMA ribbon."""
    ORDERED_BULLISH = "ordered_bullish"   # EMA10 > EMA15 > EMA20 > EMA25 > EMA30
    ORDERED_BEARISH = "ordered_bearish"   # EMA10 < EMA15 < EMA20 < EMA25 < EMA30
    TRANSITIONAL = "transitional"         # Neither fully ordered


class EMACalculator:
    """
    Single EMA calculator with warmup tracking.

    Uses standard EMA formula:
        EMA_t = price * k + EMA_{t-1} * (1-k)
        where k = 2 / (period + 1)
    """

    def __init__(self, period: int):
        self.period = period
        self.k = 2.0 / (period + 1)
        self._value: Optional[float] = None
        self._count = 0

    def update(self, price: float) -> float:
        """Update EMA with new price, return current value."""
        if self._value is None:
            self._value = price
        else:
            self._value = price * self.k + self._value * (1 - self.k)
        self._count += 1
        return self._value

    @property
    def value(self) -> Optional[float]:
        """Current EMA value."""
        return self._value

    @property
    def is_warm(self) -> bool:
        """True if we have enough bars for a stable EMA (>= period bars)."""
        return self._count >= self.period

    def reset(self):
        """Reset the calculator (call at start of new session)."""
        self._value = None
        self._count = 0


class EMARibbon:
    """
    EMA ribbon indicator using 5 EMAs on 5-minute bars.

    Tracks:
    - Current state (ordered bullish, ordered bearish, or transitional)
    - Time spent in current ordered state (state age in bars)
    - Spread between fastest and slowest EMA
    """

    def __init__(self):
        self.emas = {p: EMACalculator(p) for p in EMA_PERIODS}
        self._current_state: RibbonState = RibbonState.TRANSITIONAL
        self._state_age: int = 0

    def update(self, bar5m: Bar5m) -> dict[int, float]:
        """
        Update all EMAs with new 5-minute bar close.

        Returns:
            Dictionary of period -> EMA value
        """
        values = {p: ema.update(bar5m.close) for p, ema in self.emas.items()}

        # Update state tracking
        new_state = self._compute_state(values)
        if new_state == self._current_state:
            self._state_age += 1
        else:
            self._current_state = new_state
            self._state_age = 1

        return values

    def get_values(self) -> dict[int, Optional[float]]:
        """Return current EMA values."""
        return {p: ema.value for p, ema in self.emas.items()}

    @property
    def is_warm(self) -> bool:
        """All EMAs have enough data for stable values."""
        return all(ema.is_warm for ema in self.emas.values())

    @property
    def state(self) -> RibbonState:
        """Current ribbon state."""
        return self._current_state

    @property
    def state_age(self) -> int:
        """Number of bars in current state."""
        return self._state_age

    def is_ordered_bullish(self) -> bool:
        """EMA10 > EMA15 > EMA20 > EMA25 > EMA30 (bullish stacking)."""
        return self._current_state == RibbonState.ORDERED_BULLISH

    def is_ordered_bearish(self) -> bool:
        """EMA10 < EMA15 < EMA20 < EMA25 < EMA30 (bearish stacking)."""
        return self._current_state == RibbonState.ORDERED_BEARISH

    def is_ordered(self) -> bool:
        """True if ribbon is in either ordered state."""
        return self._current_state in (RibbonState.ORDERED_BULLISH, RibbonState.ORDERED_BEARISH)

    def spread_pct(self, price: float) -> float:
        """
        Width of ribbon as percentage of price.

        Returns (max EMA - min EMA) / price * 100
        """
        vals = self.get_values()
        if None in vals.values() or price <= 0:
            return 0.0
        values = list(vals.values())
        return (max(values) - min(values)) / price * 100

    def reset(self):
        """Reset for new session."""
        for ema in self.emas.values():
            ema.reset()
        self._current_state = RibbonState.TRANSITIONAL
        self._state_age = 0

    def _compute_state(self, values: dict[int, float]) -> RibbonState:
        """Determine ribbon state from EMA values."""
        if None in values.values():
            return RibbonState.TRANSITIONAL

        # Check bullish: 10 > 15 > 20 > 25 > 30
        if (values[10] > values[15] > values[20] > values[25] > values[30]):
            return RibbonState.ORDERED_BULLISH

        # Check bearish: 10 < 15 < 20 < 25 < 30
        if (values[10] < values[15] < values[20] < values[25] < values[30]):
            return RibbonState.ORDERED_BEARISH

        return RibbonState.TRANSITIONAL
