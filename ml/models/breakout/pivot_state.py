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
        self.latest_high: Optional[PivotPoint] = None
        self.latest_low: Optional[PivotPoint] = None
        self.all_pivots: list[PivotPoint] = []

    def update(self, pivot: PivotPoint) -> None:
        """Record a confirmed pivot from MDS."""
        self.all_pivots.append(pivot)
        if pivot.direction == "high":
            self.latest_high = pivot
        elif pivot.direction == "low":
            self.latest_low = pivot

    @property
    def high_price(self) -> Optional[float]:
        return self.latest_high.price if self.latest_high else None

    @property
    def low_price(self) -> Optional[float]:
        return self.latest_low.price if self.latest_low else None

    def high_age_minutes(self, now: datetime) -> int:
        """Minutes since the pivot high bar. now must be timezone-aware UTC."""
        if self.latest_high is None:
            return 0
        return int((now - self.latest_high.pivot_ts).total_seconds() / 60)

    def low_age_minutes(self, now: datetime) -> int:
        """Minutes since the pivot low bar. now must be timezone-aware UTC."""
        if self.latest_low is None:
            return 0
        return int((now - self.latest_low.pivot_ts).total_seconds() / 60)

    def clear(self) -> None:
        self.latest_high = None
        self.latest_low = None
        self.all_pivots.clear()
