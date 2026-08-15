# Spec: Breakout Detector Refactor
**Date:** 2026-04-10
**Status:** Ready for implementation
**Prerequisite reading:**
- `docs/decisions/subscribe_with_backfill.md`
- `docs/specs/mds_indicator_publication.md`
- `docs/decisions/adr_bar_data_source.md`

---

## Goal

Refactor `detector_service.py` to consume EMA ribbon and ATR from MDS
via ZMQ subscription instead of computing them locally. MDS is now the
single source of truth for all indicators.

This eliminates:
- `ema_ribbon.py` — local EMA computation
- `atr_calculator.py` — local ATR computation
- `bar_aggregator.py` — local 1m→5m aggregation (MDS publishes 5m bars)
- `VolumeAverageCalculator` — local volume average computation

---

## What Changes

### 1. Topic corrections

```python
# Before (wrong prefixes)
BAR_TOPIC_PREFIX = "equity.bar.1m."
CAL_TOPIC_PREFIX = "calendar."

# After
BAR_1M_TOPIC_PREFIX  = "md.equity.bar.1m."
BAR_5M_TOPIC_PREFIX  = "md.equity.bar.5m."
IND_EMA_TOPIC_PREFIX = "md.equity.indicator.ema.5m."
IND_ATR_TOPIC_PREFIX = "md.equity.indicator.atr.5m."
CAL_TOPIC_PREFIX     = "cal."
```

### 2. Remove local indicator computation

Remove from imports and all usage:
- `EMARibbon` from `ema_ribbon.py`
- `ATRCalculator` from `atr_calculator.py`
- `VolumeAverageCalculator` from `atr_calculator.py`
- `Bar5mAggregator` from `bar_aggregator.py`

Remove from `BreakoutDetector`:
- `self.aggregators` dict
- `self.ribbons` dict
- `self.atr_calcs` dict
- `self.volume_calcs` dict
- `_update_indicators()` method
- All calls to `ribbon.update()`, `atr_calc.update()`, etc.

### 3. Replace with MDS indicator state

Add per-symbol indicator state received from MDS:

```python
@dataclass
class IndicatorState:
    """Latest indicator values from MDS for one symbol."""
    ema10:        Optional[float] = None
    ema15:        Optional[float] = None
    ema20:        Optional[float] = None
    ema25:        Optional[float] = None
    ema30:        Optional[float] = None
    ribbon_state: str = "WARMING"        # BULLISH_ALIGNED, BEARISH_ALIGNED, MIXED, WARMING
    ema_warm:     bool = False
    atr:          Optional[float] = None
    atr_warm:     bool = False
    bar_index:    int = 0
    bar_time:     Optional[datetime] = None
    seq:          int = 0
```

Add to `BreakoutDetector`:
```python
self.indicators: dict[str, IndicatorState] = {}
```

### 4. Subscribe to MDS indicator topics

In `_setup_zmq()`, subscribe to:
- `md.equity.bar.5m.{SYMBOL}` — 5m bars (for level tracking)
- `md.equity.indicator.ema.5m.{SYMBOL}` — EMA ribbon
- `md.equity.indicator.atr.5m.{SYMBOL}` — ATR

Remove subscription to `md.equity.bar.1m.{SYMBOL}` — the detector
no longer processes 1m bars.

### 5. Use subscribe_with_backfill for indicators

At startup, after subscribing to indicator topics (step 1 per ADR),
call `subscribe_with_backfill` RPC for each symbol to seed indicator
state:

```python
def _warmup_indicators(self):
    for symbol in self.symbols:
        try:
            # SUB already subscribed (done in _setup_zmq)
            snapshot = mds_client.subscribe_with_backfill(
                self._mds_router_url, symbol, "indicators"
            )
            if snapshot and snapshot.get("ok"):
                snap = snapshot["snapshot"]
                state = self.indicators[symbol]
                ema = snap.get("ema", {})
                atr = snap.get("atr", {})
                state.ema10        = ema.get("ema10")
                state.ema15        = ema.get("ema15")
                state.ema20        = ema.get("ema20")
                state.ema25        = ema.get("ema25")
                state.ema30        = ema.get("ema30")
                state.ribbon_state = ema.get("ribbon_state", "WARMING")
                state.ema_warm     = ema.get("warm", False)
                state.atr          = atr.get("atr")
                state.atr_warm     = atr.get("warm", False)
                state.seq          = snapshot.get("seq", 0)
                log.info(f"{symbol} indicator snapshot: ribbon={state.ribbon_state} "
                         f"warm={state.ema_warm} atr={state.atr:.4f if state.atr else 'N/A'}")
        except Exception as e:
            log.warning(f"Failed to seed indicators for {symbol}: {e}")
```

### 6. Handle incoming indicator messages

Add handlers for EMA and ATR messages:

```python
def _handle_ema(self, symbol: str, payload: str):
    """Update EMA ribbon state from MDS."""
    try:
        data = json.loads(payload)
        seq = data.get("seq", 0)
        state = self.indicators[symbol]
        if seq <= state.seq:
            return  # dedup per subscribe_with_backfill ADR
        state.seq          = seq
        state.ema10        = data.get("ema10")
        state.ema15        = data.get("ema15")
        state.ema20        = data.get("ema20")
        state.ema25        = data.get("ema25")
        state.ema30        = data.get("ema30")
        state.ribbon_state = data.get("ribbon_state", "WARMING")
        state.ema_warm     = data.get("warm", False)
        state.bar_index    = data.get("bar_index", 0)
        state.bar_time     = datetime.fromisoformat(
            data["bar_time"].replace("Z", "+00:00")
        ) if "bar_time" in data else None
    except Exception as e:
        log.warning(f"Error processing EMA for {symbol}: {e}")

def _handle_atr(self, symbol: str, payload: str):
    """Update ATR state from MDS."""
    try:
        data = json.loads(payload)
        seq = data.get("seq", 0)
        state = self.indicators[symbol]
        # ATR shares seq with EMA — use max to avoid regression
        state.seq      = max(state.seq, seq)
        state.atr      = data.get("atr")
        state.atr_warm = data.get("warm", False)
    except Exception as e:
        log.warning(f"Error processing ATR for {symbol}: {e}")
```

### 7. Handle 5m bar messages

Subscribe to `md.equity.bar.5m.{SYMBOL}` instead of aggregating
locally from 1m bars. When a 5m bar arrives, update `LevelTracker`
and check for breakouts.

```python
def _handle_bar_5m(self, symbol: str, payload: str):
    """Process completed 5m bar from MDS."""
    try:
        data = json.loads(payload)
        bar_data = data["data"]

        # Skip non-regular session
        session = bar_data.get("session", "regular")
        if session != "regular":
            return

        ts = datetime.fromisoformat(
            bar_data["ts"].replace("Z", "+00:00")
        ).astimezone(NY)

        bar5m = Bar5m(
            ts         = ts,
            open       = float(bar_data["open"]),
            high       = float(bar_data["high"]),
            low        = float(bar_data["low"]),
            close      = float(bar_data["close"]),
            volume     = int(bar_data["volume"]),
            vwap       = float(bar_data.get("vwap", bar_data["close"])),
        )

        # Update level tracker
        self.levels[symbol].update(bar5m)

        # Track session open
        if self.session_open[symbol] is None:
            self.session_open[symbol] = bar5m.open
            if self.prior_close[symbol] is not None:
                self.gap_pct[symbol] = (
                    (bar5m.open - self.prior_close[symbol])
                    / self.prior_close[symbol]
                )

        # Check for breakout using MDS indicators
        self._check_breakout(symbol, bar5m)

    except Exception as e:
        log.warning(f"Error processing 5m bar for {symbol}: {e}")
```

### 8. Update `_check_breakout` to use MDS indicators

```python
def _check_breakout(self, symbol: str, bar5m: Bar5m):
    """Check for breakout conditions using MDS indicator state."""
    ind = self.indicators[symbol]
    levels = self.levels[symbol]

    # Need warm indicators from MDS
    if not ind.ema_warm or not ind.atr_warm:
        log.debug(f"{symbol} indicators not warm: ema={ind.ema_warm} atr={ind.atr_warm}")
        return

    if ind.atr is None or ind.atr <= 0:
        return

    atr = ind.atr
    avg_vol = self.volume_calcs[symbol].value if self.volume_calcs else None
    volume_ratio = bar5m.volume / avg_vol if avg_vol and avg_vol > 0 else 1.0

    signal_config = SignalConfig(
        level_age_threshold    = self.config.level_age_threshold,
        ribbon_age_threshold   = self.config.ribbon_age_threshold,
        ribbon_spread_min_pct  = self.config.ribbon_spread_min_pct,
        break_bar_atr_min      = self.config.break_bar_atr_min,
        break_bar_close_pct    = self.config.break_bar_close_pct,
        volume_ratio_min       = self.config.volume_ratio_min,
        clear_air_atr_min      = self.config.clear_air_atr_min,
        gap_atr_threshold      = self.config.gap_atr_threshold,
    )
    checker = BreakoutConditionChecker(signal_config)

    # Convert MDS ribbon_state string to RibbonState enum
    from ml.models.breakout.ema_ribbon import RibbonState
    try:
        ribbon_state = RibbonState(ind.ribbon_state)
    except ValueError:
        ribbon_state = RibbonState.WARMING

    # Check long breakout
    if levels.high_price is not None:
        candidate = checker.check_long_breakout(
            symbol             = symbol,
            bar5m              = bar5m,
            level_price        = levels.high_price,
            level_age_minutes  = levels.high_age_minutes(),
            ribbon_state       = ribbon_state,
            ribbon_age         = ind.bar_index,   # proxy for ribbon age
            ribbon_spread_pct  = _ribbon_spread_pct(ind, bar5m.close),
            atr                = atr,
            volume_ratio       = volume_ratio,
            gap_pct            = self.gap_pct[symbol],
            prior_session_high = self.prior_session_high[symbol],
        )
        if candidate:
            self._handle_signal(candidate)

    # Check short breakout
    if levels.low_price is not None:
        candidate = checker.check_short_breakout(
            symbol            = symbol,
            bar5m             = bar5m,
            level_price       = levels.low_price,
            level_age_minutes = levels.low_age_minutes(),
            ribbon_state      = ribbon_state,
            ribbon_age        = ind.bar_index,
            ribbon_spread_pct = _ribbon_spread_pct(ind, bar5m.close),
            atr               = atr,
            volume_ratio      = volume_ratio,
            gap_pct           = self.gap_pct[symbol],
            prior_session_low = self.prior_session_low[symbol],
        )
        if candidate:
            self._handle_signal(candidate)


def _ribbon_spread_pct(ind: IndicatorState, close: float) -> float:
    """Compute ribbon spread % from MDS indicator state."""
    if ind.ema10 is None or ind.ema30 is None or close <= 0:
        return 0.0
    return abs(ind.ema10 - ind.ema30) / close
```

### 9. Update `_handle_message` routing

```python
def _handle_message(self, topic: str, payload: str):
    if topic.startswith(BAR_5M_TOPIC_PREFIX):
        symbol = topic[len(BAR_5M_TOPIC_PREFIX):]
        if symbol in self.symbols:
            self._handle_bar_5m(symbol, payload)
    elif topic.startswith(IND_EMA_TOPIC_PREFIX):
        symbol = topic[len(IND_EMA_TOPIC_PREFIX):]
        if symbol in self.symbols:
            self._handle_ema(symbol, payload)
    elif topic.startswith(IND_ATR_TOPIC_PREFIX):
        symbol = topic[len(IND_ATR_TOPIC_PREFIX):]
        if symbol in self.symbols:
            self._handle_atr(symbol, payload)
    elif topic.startswith(CAL_TOPIC_PREFIX):
        self._handle_calendar(topic, payload)
```

### 10. Update `_reset_session_state`

Remove resets for deleted state objects:
```python
def _reset_session_state(self):
    self._today = datetime.now(NY).date()
    self._signals_today.clear()

    for symbol in self.symbols:
        self.levels[symbol].clear()
        self.indicators[symbol] = IndicatorState()  # reset MDS indicator state
        self.session_open[symbol] = None
        self.gap_pct[symbol] = 0.0

    self._fetch_prior_session_data()
    # Re-seed indicators via subscribe_with_backfill
    self._warmup_indicators()
```

### 11. Startup ordering (critical — per subscribe_with_backfill ADR)

```
1. _setup_zmq() — subscribe to ALL topics FIRST (before any RPC)
2. _fetch_prior_session_data() — get prior session close/high/low
3. _warmup_levels() — fetch today's 5m bars, seed LevelTracker
4. _warmup_indicators() — subscribe_with_backfill for each symbol
5. Set self._today
6. Start bar ingestion thread
```

Note: `_warmup_levels()` replaces the current `_warmup_from_history()`.
It fetches today's 5m bars from ClickHouse (or MDS get_bars), replays
them through `LevelTracker` only. Indicator state comes entirely from
MDS via subscribe_with_backfill — do not recompute locally.

---

## Files to Delete

Once the refactor is complete and verified:
- `ml/models/breakout/ema_ribbon.py`
- `ml/models/breakout/atr_calculator.py`
- `ml/models/breakout/bar_aggregator.py`

**Do not delete during implementation** — keep until verified working.
Delete in a follow-up commit after a full session runs clean.

---

## Files Unchanged

- `ml/models/breakout/level_tracker.py` — pure logic, no indicator deps
- `ml/models/breakout/signal_generator.py` — pure logic, no indicator deps

---

## VolumeAverageCalculator

`VolumeAverageCalculator` is used for volume ratio in breakout checks.
MDS does not publish volume averages. Two options:

1. **Keep it locally** — feed volume from 5m bar payloads as they arrive
2. **Remove it** — pass `volume_ratio = 1.0` as a neutral value

Recommendation: **keep it locally** since it's a simple rolling average
with no formula dependency on MDS. Just feed `bar5m.volume` from the
5m bar payload. This is different from EMA/ATR which have complex
seeding requirements.

Keep `VolumeAverageCalculator` and `self.volume_calcs` dict.
Update it from `_handle_bar_5m` with `bar5m.volume`.

---

## mds_client additions needed

`subscribe_with_backfill` RPC call needs to be added to
`ml/shared/mds_client.py`. The RPC format per the ADR:

```python
def subscribe_with_backfill(router_url: str, symbol: str,
                            data_type: str, timeframe: str = "5m") -> dict:
    """
    Call MDS subscribe_with_backfill RPC.
    Returns the full response dict or None on failure.
    """
    request = {
        "op":        "subscribe_with_backfill",
        "data_type": data_type,
        "symbol":    symbol,
        "timeframe": timeframe,
    }
    # Use existing DEALER/ROUTER pattern in mds_client.py
    ...
```

---

## Implementation Order

1. Add `subscribe_with_backfill` to `mds_client.py`
2. Add `IndicatorState` dataclass to `detector_service.py`
3. Update `_setup_zmq` — new subscriptions
4. Update `_init_symbol_state` — add `self.indicators`, keep `self.volume_calcs`
5. Add `_warmup_levels` (replaces `_warmup_from_history` for level seeding only)
6. Add `_warmup_indicators` (new — uses subscribe_with_backfill)
7. Add `_handle_bar_5m`, `_handle_ema`, `_handle_atr`
8. Update `_handle_message` routing
9. Update `_check_breakout` to use `IndicatorState`
10. Update `_reset_session_state`
11. Fix topic prefix constants
12. Update startup ordering
13. Test: run detector, verify signals fire correctly
14. Follow-up commit: delete `ema_ribbon.py`, `atr_calculator.py`, `bar_aggregator.py`

---

## Out of Scope

- Changes to `level_tracker.py` or `signal_generator.py`
- ClickHouse persistence for breakout candidates (existing TODO)
- ZMQ publication of signals (existing TODO, phase >= 1)
- Multi-timeframe indicator support (5m only for now)
