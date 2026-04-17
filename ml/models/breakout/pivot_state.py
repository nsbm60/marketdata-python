"""
ml/models/breakout/pivot_state.py

Holds confirmed pivot points for one symbol, received from MDS.
Replaces LevelTracker — levels are now MDS-computed pivot highs/lows
rather than running session highs/lows.
"""

from datetime import datetime
from typing import Optional

from ml.models.breakout.types import PivotPoint


class PivotState:
    """Per-symbol confirmed pivot state.

    All datetime arguments (now, pivot_ts, confirmed_ts) must be
    timezone-aware UTC datetimes.
    """

    def __init__(self):
        self.highs: list[PivotPoint] = []
        self.lows: list[PivotPoint] = []

    def update(self, pivot: PivotPoint) -> None:
        """Record a confirmed pivot from MDS."""
        if pivot.direction == "high":
            self.highs.append(pivot)
            self.highs.sort(key=lambda p: p.price, reverse=True)
        elif pivot.direction == "low":
            self.lows.append(pivot)
            self.lows.sort(key=lambda p: p.price)

    @property
    def high_levels(self) -> list[PivotPoint]:
        """All confirmed pivot highs sorted descending by price."""
        return self.highs

    @property
    def low_levels(self) -> list[PivotPoint]:
        """All confirmed pivot lows sorted ascending by price."""
        return self.lows

    @property
    def most_recent_high(self) -> Optional[PivotPoint]:
        if not self.highs:
            return None
        return max(self.highs, key=lambda p: p.confirmed_ts)

    @property
    def most_recent_low(self) -> Optional[PivotPoint]:
        if not self.lows:
            return None
        return max(self.lows, key=lambda p: p.confirmed_ts)

    def clear(self) -> None:
        self.highs.clear()
        self.lows.clear()
