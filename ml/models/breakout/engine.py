"""
ml/models/breakout/engine.py

Core breakout detection logic — timeframe and data-source agnostic.

Takes bars and indicator state as input. Does not know whether data
comes from MDS ZMQ or ClickHouse. Both live and historical paths use
this class identically.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ml.models.breakout.pivot_state import PivotState
from ml.models.breakout.signal_generator import (
    BreakoutCandidate,
    BreakoutConditionChecker,
    SignalConfig,
)
from ml.models.breakout.types import Bar, PivotPoint, RibbonState, VolumeAverageCalculator

log = logging.getLogger(__name__)


# MDS publishes "BULLISH_ALIGNED" etc.; local signal_generator uses RibbonState enum
# with different value strings. This map bridges the two.
_MDS_RIBBON_MAP = {
    "BULLISH_ALIGNED": RibbonState.ORDERED_BULLISH,
    "BEARISH_ALIGNED": RibbonState.ORDERED_BEARISH,
    "MIXED":           RibbonState.TRANSITIONAL,
    "WARMING":         RibbonState.TRANSITIONAL,
}


@dataclass
class IndicatorState:
    """Latest indicator values for one symbol."""
    ema10:        Optional[float] = None
    ema15:        Optional[float] = None
    ema20:        Optional[float] = None
    ema25:        Optional[float] = None
    ema30:        Optional[float] = None
    ribbon_state: str = "WARMING"
    ema_warm:     bool = False
    atr:          Optional[float] = None
    atr_warm:     bool = False
    bar_index:    int = 0
    bar_time:     Optional[datetime] = None
    seq:          int = 0


def ribbon_spread_pct(ind: IndicatorState, close: float) -> float:
    """Compute ribbon spread % from indicator state."""
    if ind.ema10 is None or ind.ema30 is None or close <= 0:
        return 0.0
    return abs(ind.ema10 - ind.ema30) / close


class BreakoutEngine:
    """
    Core breakout detection logic — timeframe and data-source agnostic.

    Owns all per-symbol mutable state: PivotState, VolumeAverageCalculator,
    IndicatorState, session open/gap.  Processes bars one at a time and returns
    any breakout candidates detected.

    Both the live detector (ZMQ) and historical replay (ClickHouse) use this
    class identically — they differ only in how bars and indicators are fed in.
    """

    def __init__(self, timeframe: str, config: SignalConfig):
        """
        Args:
            timeframe: Period string, e.g. "1m", "5m", "60m"
            config: Signal thresholds for breakout detection
        """
        self.timeframe = timeframe
        self.config = config

        # Per-symbol state — populated by init_symbols()
        self.pivots: dict[str, PivotState] = {}
        self.volume_calcs: dict[str, VolumeAverageCalculator] = {}
        self.indicators: dict[str, IndicatorState] = {}
        self.session_open: dict[str, Optional[float]] = {}
        self.gap_pct: dict[str, float] = {}
        self.prior_close: dict[str, Optional[float]] = {}
        self.prior_session_high: dict[str, Optional[float]] = {}
        self.prior_session_low: dict[str, Optional[float]] = {}
        self._bar_index: dict[str, int] = {}
        self.ribbon_age: dict[str, int] = {}
        self.last_ribbon_state: dict[str, str] = {}
        self.prior_bar_close: dict[str, Optional[float]] = {}

    def init_symbols(self, symbols: list[str]) -> None:
        """Initialize per-symbol state for given symbols."""
        for symbol in symbols:
            self.indicators[symbol] = IndicatorState()
            self.pivots[symbol] = PivotState()
            self.volume_calcs[symbol] = VolumeAverageCalculator()
            self.session_open[symbol] = None
            self.gap_pct[symbol] = 0.0
            self.prior_close[symbol] = None
            self.prior_session_high[symbol] = None
            self.prior_session_low[symbol] = None
            self._bar_index[symbol] = 0
            self.ribbon_age[symbol] = 0
            self.last_ribbon_state[symbol] = ""
            self.prior_bar_close[symbol] = None

    def set_prior_session(
        self, symbol: str, close: float,
        high: float, low: float,
    ) -> None:
        """Seed prior session close/high/low for gap calculation."""
        self.prior_close[symbol] = close
        self.prior_session_high[symbol] = high
        self.prior_session_low[symbol] = low

    def update_pivot(self, symbol: str, pivot: PivotPoint) -> None:
        """Record a confirmed pivot from MDS or ClickHouse replay."""
        state = self.pivots.get(symbol)
        if state is not None:
            state.update(pivot)

    def update_indicator(
        self,
        symbol: str,
        *,
        ema_dict: Optional[dict] = None,
        atr_value: Optional[float] = None,
        ribbon_state: Optional[str] = None,
        ema_warm: Optional[bool] = None,
        atr_warm: Optional[bool] = None,
        bar_index: Optional[int] = None,
        seq: int = 0,
    ) -> None:
        """Update indicator state from MDS payload or ClickHouse row.

        Callers pass whichever fields they have. Missing fields are left
        unchanged in the current state.
        """
        state = self.indicators.get(symbol)
        if state is None:
            return

        if seq > 0 and seq <= state.seq:
            return  # dedup per subscribe_with_backfill ADR

        if seq > 0:
            state.seq = max(state.seq, seq)

        if ema_dict is not None:
            state.ema10 = ema_dict.get("ema10", state.ema10)
            state.ema15 = ema_dict.get("ema15", state.ema15)
            state.ema20 = ema_dict.get("ema20", state.ema20)
            state.ema25 = ema_dict.get("ema25", state.ema25)
            state.ema30 = ema_dict.get("ema30", state.ema30)

        if ribbon_state is not None:
            state.ribbon_state = ribbon_state
            prev = self.last_ribbon_state.get(symbol, "")
            if ribbon_state == prev and ribbon_state in ("BULLISH_ALIGNED", "BEARISH_ALIGNED"):
                self.ribbon_age[symbol] = self.ribbon_age.get(symbol, 0) + 1
            else:
                self.ribbon_age[symbol] = 1
            self.last_ribbon_state[symbol] = ribbon_state
        if ema_warm is not None:
            state.ema_warm = ema_warm
        if atr_value is not None:
            state.atr = atr_value
        if atr_warm is not None:
            state.atr_warm = atr_warm
        if bar_index is not None:
            state.bar_index = bar_index

    def on_bar(self, symbol: str, bar: Bar) -> list[BreakoutCandidate]:
        """Process a completed bar. Returns any breakout candidates detected."""
        # Track session open from first bar
        if self.session_open.get(symbol) is None:
            self.session_open[symbol] = bar.open
            prior = self.prior_close.get(symbol)
            if prior is not None and prior > 0:
                self.gap_pct[symbol] = (bar.open - prior) / prior

        self.volume_calcs[symbol].update(bar.volume)
        self._bar_index[symbol] = self._bar_index.get(symbol, 0) + 1

        result = self._check_breakout(symbol, bar)
        self.prior_bar_close[symbol] = bar.close
        return result

    def warmup_bar(self, symbol: str, bar: Bar) -> None:
        """Replay a historical bar for volume warmup without checking breakouts."""
        if self.session_open.get(symbol) is None:
            self.session_open[symbol] = bar.open
            prior = self.prior_close.get(symbol)
            if prior is not None and prior > 0:
                self.gap_pct[symbol] = (bar.open - prior) / prior

        self.volume_calcs[symbol].update(bar.volume)
        self._bar_index[symbol] = self._bar_index.get(symbol, 0) + 1

    def reset_session(self, symbols: list[str]) -> None:
        """Reset intraday state at session boundary."""
        for symbol in symbols:
            if symbol in self.indicators:
                self.indicators[symbol] = IndicatorState()
            if symbol in self.pivots:
                self.pivots[symbol].clear()
            if symbol in self.volume_calcs:
                self.volume_calcs[symbol].reset()
            self.session_open[symbol] = None
            self.gap_pct[symbol] = 0.0
            self._bar_index[symbol] = 0
            self.ribbon_age[symbol] = 0
            self.last_ribbon_state[symbol] = ""
            self.prior_bar_close[symbol] = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_breakout(self, symbol: str, bar: Bar) -> list[BreakoutCandidate]:
        """Check for breakout conditions using current indicator state."""
        ind = self.indicators.get(symbol)
        pivots = self.pivots.get(symbol)
        vol_calc = self.volume_calcs.get(symbol)

        if ind is None or pivots is None or vol_calc is None:
            return []

        # Need warm indicators
        if not ind.ema_warm or not ind.atr_warm:
            return []
        if ind.atr is None or ind.atr <= 0:
            return []

        atr = ind.atr
        avg_vol = vol_calc.value
        volume_ratio = bar.volume / avg_vol if avg_vol and avg_vol > 0 else 1.0

        checker = BreakoutConditionChecker(self.config)
        rs = _MDS_RIBBON_MAP.get(ind.ribbon_state, RibbonState.TRANSITIONAL)

        # MDS bar_index (from update_indicator) takes precedence; local count is fallback for replay
        bar_index = ind.bar_index if ind.bar_index > 0 else self._bar_index.get(symbol, 0)

        candidates = []

        # Check long breakout — iterate high levels, lowest high first
        most_recent_high = pivots.most_recent_high
        for level in reversed(pivots.high_levels):
            # Skip if a higher pivot high exists confirmed after this level
            if any(h.price > level.price and h.confirmed_ts > level.confirmed_ts
                   for h in pivots.high_levels):
                continue
            is_most_recent = (most_recent_high is not None
                              and level.confirmed_ts == most_recent_high.confirmed_ts)
            level_age = int((bar.ts - level.pivot_ts).total_seconds() / 60)
            candidate = checker.check_long_breakout(
                symbol=symbol,
                bar=bar,
                level_price=level.price,
                level_age_minutes=level_age,
                is_most_recent=is_most_recent,
                ribbon_state=rs,
                ribbon_age=self.ribbon_age.get(symbol, 0),
                ribbon_spread_pct=ribbon_spread_pct(ind, bar.close),
                atr=atr,
                volume_ratio=volume_ratio,
                gap_pct=self.gap_pct.get(symbol, 0.0),
                prior_bar_close=self.prior_bar_close.get(symbol),
                bar_index=bar_index,
            )
            if candidate:
                candidate.prior_session_high = self.prior_session_high.get(symbol)
                candidate.prior_session_low = self.prior_session_low.get(symbol)
                candidates.append(candidate)
                break  # fire on first matching level

        # Check short breakout — iterate low levels, highest low first
        most_recent_low = pivots.most_recent_low
        for level in reversed(pivots.low_levels):
            # Skip if a lower pivot low exists confirmed after this level
            if any(l.price < level.price and l.confirmed_ts > level.confirmed_ts
                   for l in pivots.low_levels):
                continue
            is_most_recent = (most_recent_low is not None
                              and level.confirmed_ts == most_recent_low.confirmed_ts)
            level_age = int((bar.ts - level.pivot_ts).total_seconds() / 60)
            candidate = checker.check_short_breakout(
                symbol=symbol,
                bar=bar,
                level_price=level.price,
                level_age_minutes=level_age,
                is_most_recent=is_most_recent,
                ribbon_state=rs,
                ribbon_age=self.ribbon_age.get(symbol, 0),
                ribbon_spread_pct=ribbon_spread_pct(ind, bar.close),
                atr=atr,
                volume_ratio=volume_ratio,
                gap_pct=self.gap_pct.get(symbol, 0.0),
                prior_bar_close=self.prior_bar_close.get(symbol),
                bar_index=bar_index,
            )
            if candidate:
                candidate.prior_session_high = self.prior_session_high.get(symbol)
                candidate.prior_session_low = self.prior_session_low.get(symbol)
                candidates.append(candidate)
                break  # fire on first matching level

        return candidates
