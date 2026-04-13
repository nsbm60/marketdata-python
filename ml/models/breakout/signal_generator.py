"""
ml/models/breakout/signal_generator.py

Breakout signal generation and quality scoring.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from ml.models.breakout.types import Bar, RibbonState


class BreakoutDirection(Enum):
    """Direction of breakout."""
    LONG = "long"
    SHORT = "short"


@dataclass
class SignalConfig:
    """Configuration for breakout signal conditions."""
    level_age_threshold: int = 30        # Min minutes level must be aged
    ribbon_age_threshold: int = 3        # Min bars in ordered state
    ribbon_spread_min_pct: float = 0.1   # Min ribbon spread %
    break_bar_atr_min: float = 0.5       # Min bar range in ATR units
    break_bar_close_pct: float = 0.6     # Bar must close in top/bottom X%
    volume_ratio_min: float = 1.0        # Min volume vs average
    clear_air_atr_min: float = 0.5       # Min distance to prior session level
    gap_atr_threshold: float = 2.0       # Max gap in ATR units


@dataclass
class BreakoutCandidate:
    """A detected breakout signal."""
    symbol: str
    direction: BreakoutDirection
    ts: datetime
    price: float
    level_price: float
    level_age_minutes: int
    ribbon_state: RibbonState
    ribbon_age: int
    ribbon_spread_pct: float
    atr: float
    bar_range_atr: float
    bar_close_pct: float
    volume_ratio: float
    gap_pct: float
    bar_index: int
    score: float


class BreakoutConditionChecker:
    """
    Checks breakout conditions and generates signals.

    Conditions for a valid breakout:
    1. Level age: high/low must be aged (not just established)
    2. Ribbon alignment: EMAs in ordered state for N bars
    3. Ribbon spread: spread must indicate strong trend
    4. Break bar quality: bar range and close position
    5. Volume confirmation: volume above average
    6. Clear air: distance to prior session level
    7. Gap filter: avoid large gap days
    """

    def __init__(self, config: SignalConfig):
        self.config = config

    def check_long_breakout(
        self,
        symbol: str,
        bar: Bar,
        level_price: float,
        level_age_minutes: int,
        ribbon_state: RibbonState,
        ribbon_age: int,
        ribbon_spread_pct: float,
        atr: float,
        volume_ratio: float,
        gap_pct: float,
        prior_session_high: Optional[float],
        bar_index: int = 0,
    ) -> Optional[BreakoutCandidate]:
        """
        Check for long breakout (new high).

        Returns BreakoutCandidate if all conditions met, None otherwise.
        """
        c = self.config

        # Must make new high
        if bar.high <= level_price:
            return None

        # Level must be aged
        if level_age_minutes < c.level_age_threshold:
            return None

        # Ribbon must be bullish and mature
        if ribbon_state != RibbonState.ORDERED_BULLISH:
            return None
        if ribbon_age < c.ribbon_age_threshold:
            return None

        # Ribbon spread check
        if ribbon_spread_pct < c.ribbon_spread_min_pct:
            return None

        # Bar quality
        bar_range = bar.high - bar.low
        bar_range_atr = bar_range / atr if atr > 0 else 0
        if bar_range_atr < c.break_bar_atr_min:
            return None

        # Close in upper portion of bar
        if bar_range > 0:
            bar_close_pct = (bar.close - bar.low) / bar_range
        else:
            bar_close_pct = 0.5
        if bar_close_pct < c.break_bar_close_pct:
            return None

        # Volume confirmation
        if volume_ratio < c.volume_ratio_min:
            return None

        # Gap filter
        if atr > 0 and abs(gap_pct * bar.close) / atr > c.gap_atr_threshold:
            return None

        # Clear air above prior session high
        if prior_session_high is not None and atr > 0:
            distance_to_prior = bar.high - prior_session_high
            if distance_to_prior < 0:  # Below prior high = no clear air
                return None
            if distance_to_prior / atr < c.clear_air_atr_min:
                return None

        # Calculate score
        score = self._calculate_score(
            level_age_minutes, ribbon_age, ribbon_spread_pct,
            bar_range_atr, bar_close_pct, volume_ratio
        )

        return BreakoutCandidate(
            symbol=symbol,
            direction=BreakoutDirection.LONG,
            ts=bar.ts,
            price=bar.close,
            level_price=level_price,
            level_age_minutes=level_age_minutes,
            ribbon_state=ribbon_state,
            ribbon_age=ribbon_age,
            ribbon_spread_pct=ribbon_spread_pct,
            atr=atr,
            bar_range_atr=bar_range_atr,
            bar_close_pct=bar_close_pct,
            volume_ratio=volume_ratio,
            gap_pct=gap_pct,
            bar_index=bar_index,
            score=score,
        )

    def check_short_breakout(
        self,
        symbol: str,
        bar: Bar,
        level_price: float,
        level_age_minutes: int,
        ribbon_state: RibbonState,
        ribbon_age: int,
        ribbon_spread_pct: float,
        atr: float,
        volume_ratio: float,
        gap_pct: float,
        prior_session_low: Optional[float],
        bar_index: int = 0,
    ) -> Optional[BreakoutCandidate]:
        """
        Check for short breakout (new low).

        Returns BreakoutCandidate if all conditions met, None otherwise.
        """
        c = self.config

        # Must make new low
        if bar.low >= level_price:
            return None

        # Level must be aged
        if level_age_minutes < c.level_age_threshold:
            return None

        # Ribbon must be bearish and mature
        if ribbon_state != RibbonState.ORDERED_BEARISH:
            return None
        if ribbon_age < c.ribbon_age_threshold:
            return None

        # Ribbon spread check
        if ribbon_spread_pct < c.ribbon_spread_min_pct:
            return None

        # Bar quality
        bar_range = bar.high - bar.low
        bar_range_atr = bar_range / atr if atr > 0 else 0
        if bar_range_atr < c.break_bar_atr_min:
            return None

        # Close in lower portion of bar
        if bar_range > 0:
            bar_close_pct = (bar.high - bar.close) / bar_range  # Inverted for shorts
        else:
            bar_close_pct = 0.5
        if bar_close_pct < c.break_bar_close_pct:
            return None

        # Volume confirmation
        if volume_ratio < c.volume_ratio_min:
            return None

        # Gap filter
        if atr > 0 and abs(gap_pct * bar.close) / atr > c.gap_atr_threshold:
            return None

        # Clear air below prior session low
        if prior_session_low is not None and atr > 0:
            distance_to_prior = prior_session_low - bar.low
            if distance_to_prior < 0:  # Above prior low = no clear air
                return None
            if distance_to_prior / atr < c.clear_air_atr_min:
                return None

        # Calculate score
        score = self._calculate_score(
            level_age_minutes, ribbon_age, ribbon_spread_pct,
            bar_range_atr, bar_close_pct, volume_ratio
        )

        return BreakoutCandidate(
            symbol=symbol,
            direction=BreakoutDirection.SHORT,
            ts=bar.ts,
            price=bar.close,
            level_price=level_price,
            level_age_minutes=level_age_minutes,
            ribbon_state=ribbon_state,
            ribbon_age=ribbon_age,
            ribbon_spread_pct=ribbon_spread_pct,
            atr=atr,
            bar_range_atr=bar_range_atr,
            bar_close_pct=bar_close_pct,
            volume_ratio=volume_ratio,
            gap_pct=gap_pct,
            bar_index=bar_index,
            score=score,
        )

    def _calculate_score(
        self,
        level_age: int,
        ribbon_age: int,
        ribbon_spread: float,
        bar_range_atr: float,
        bar_close_pct: float,
        volume_ratio: float,
    ) -> float:
        """
        Calculate quality score (0-100).

        Higher scores indicate higher quality breakouts.
        """
        score = 0.0

        # Level age component (0-25 points)
        # More aged = better, max at 60 minutes
        level_score = min(level_age / 60.0, 1.0) * 25
        score += level_score

        # Ribbon age component (0-20 points)
        # More mature ribbon = better, max at 10 bars
        ribbon_score = min(ribbon_age / 10.0, 1.0) * 20
        score += ribbon_score

        # Ribbon spread component (0-15 points)
        # Wider spread = stronger trend, max at 0.5%
        spread_score = min(ribbon_spread / 0.5, 1.0) * 15
        score += spread_score

        # Bar quality component (0-20 points)
        bar_score = min(bar_range_atr / 1.5, 1.0) * 10
        bar_score += bar_close_pct * 10
        score += bar_score

        # Volume component (0-20 points)
        # Higher volume = better, max at 2x average
        vol_score = min(volume_ratio / 2.0, 1.0) * 20
        score += vol_score

        return round(score, 1)
