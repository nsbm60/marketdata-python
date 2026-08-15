# Spec: Multi-Timeframe Breakout Detector + Historical Replay
**Date:** 2026-04-12
**Status:** Ready for implementation

---

## Overview

Two related workstreams:

1. **Multi-timeframe live detector** — refactor `detector_service.py` to run
   simultaneously across all 7 timeframes (1m, 5m, 10m, 15m, 20m, 30m, 60m)

2. **Historical replay script** — run the same detection logic against
   ClickHouse historical data to populate `breakout_candidates` back to
   2020-12-31

Both share the same core detection logic. The key architectural decision is
to extract that logic into a shared `BreakoutEngine` so it's not duplicated.

---

## Shared Architecture: BreakoutEngine

Extract the per-bar detection logic from `detector_service.py` into a new
module `ml/models/breakout/engine.py`:

```python
class BreakoutEngine:
    """
    Core breakout detection logic — timeframe and data-source agnostic.
    
    Takes bars and indicator state as input. Does not know whether data
    comes from MDS ZMQ or ClickHouse. Both live and historical paths use
    this class identically.
    """
    
    def __init__(self, timeframe: str, config: BreakoutConfig):
        self.timeframe = timeframe
        self.config = config
        # Per-symbol state
        self.levels: dict[str, LevelTracker] = {}
        self.volume_calcs: dict[str, VolumeAverageCalculator] = {}
        self.indicators: dict[str, IndicatorState] = {}
        self.session_open: dict[str, Optional[float]] = {}
        self.gap_pct: dict[str, float] = {}
        self.prior_close: dict[str, Optional[float]] = {}
        self.prior_session_high: dict[str, Optional[float]] = {}
        self.prior_session_low: dict[str, Optional[float]] = {}
    
    def init_symbols(self, symbols: list[str]) -> None:
        """Initialize per-symbol state for given symbols."""
        ...
    
    def set_prior_session(self, symbol: str, close: float,
                          high: float, low: float) -> None:
        """Seed prior session data before replay begins."""
        ...
    
    def update_indicator(self, symbol: str, ema: dict,
                         atr: dict) -> None:
        """Update indicator state from MDS payload or ClickHouse row."""
        ...
    
    def on_bar(self, symbol: str, bar: Bar,
               session_date: date) -> list[BreakoutCandidate]:
        """
        Process a completed bar. Returns any breakout candidates detected.
        Updates LevelTracker, VolumeAverageCalculator, checks breakouts.
        """
        ...
    
    def reset_session(self, symbols: list[str]) -> None:
        """Reset intraday state at session boundary."""
        ...
```

---

## Part 1: Multi-Timeframe Live Detector

### Design

Run one `BreakoutEngine` per timeframe, all within a single process:

```python
TIMEFRAMES = ["1m", "5m", "10m", "15m", "20m", "30m", "60m"]

self.engines: dict[str, BreakoutEngine] = {
    tf: BreakoutEngine(tf, config) for tf in TIMEFRAMES
}
```

### ZMQ Subscriptions

For each timeframe and symbol, subscribe to:
- `md.equity.bar.{TF}.{SYMBOL}` — completed bars
- `md.equity.indicator.ema.{TF}.{SYMBOL}` — EMA ribbon
- `md.equity.indicator.atr.{TF}.{SYMBOL}` — ATR

Topic prefix constants become dynamic:
```python
def bar_topic(tf: str, symbol: str) -> str:
    return f"md.equity.bar.{tf}.{symbol}"

def ema_topic(tf: str, symbol: str) -> str:
    return f"md.equity.indicator.ema.{tf}.{symbol}"

def atr_topic(tf: str, symbol: str) -> str:
    return f"md.equity.indicator.atr.{tf}.{symbol}"
```

### Message Routing

`_handle_message` routes by topic prefix to the correct engine:

```python
def _handle_message(self, topic: str, payload: str):
    for tf in TIMEFRAMES:
        if topic.startswith(f"md.equity.bar.{tf}."):
            symbol = topic[len(f"md.equity.bar.{tf}."):]
            if symbol in self.symbols:
                bar = parse_bar(payload)
                candidates = self.engines[tf].on_bar(symbol, bar, today)
                for c in candidates:
                    self._handle_signal(c)
            return
        if topic.startswith(f"md.equity.indicator.ema.{tf}."):
            symbol = topic[len(f"md.equity.indicator.ema.{tf}."):]
            if symbol in self.symbols:
                self.engines[tf].update_indicator_ema(symbol, payload)
            return
        if topic.startswith(f"md.equity.indicator.atr.{tf}."):
            symbol = topic[len(f"md.equity.indicator.atr.{tf}."):]
            if symbol in self.symbols:
                self.engines[tf].update_indicator_atr(symbol, payload)
            return
    if topic.startswith(CAL_TOPIC_PREFIX):
        self._handle_calendar(topic, payload)
```

### Startup

For each timeframe, subscribe to ZMQ topics FIRST, then:
1. `_warmup_levels(tf)` — fetch today's bars via `get_bars(period=tf)`
2. `_warmup_indicators(tf)` — `subscribe_with_backfill` per symbol

### Session Reset

On market open, reset all engines:
```python
for engine in self.engines.values():
    engine.reset_session(self.symbols)
```

---

## Part 2: Historical Replay Script

### Entry Point

```
python tools/replay_breakouts.py \
    --start-date 2020-12-31 \
    --end-date 2026-04-11 \
    --symbols NVDA,AMD,...  # optional, default: TradingUniverse
    --timeframes 1m,5m,10m,15m,20m,30m,60m  # optional, default: all
```

### Algorithm

```
for each trading day from start_date to end_date:
    load prior session data from ClickHouse (close, high, low per symbol)
    
    for each timeframe:
        engine = BreakoutEngine(timeframe, config)
        engine.init_symbols(symbols)
        
        for each symbol:
            engine.set_prior_session(symbol, prior_close, prior_high, prior_low)
        
        for each symbol:
            bars = query stock_bar WHERE period=tf AND session=regular AND date=day
            indicators = query indicator WHERE timeframe=tf AND session=day
            
            # Merge bars and indicators by timestamp, replay in order
            for bar, indicator in zip(bars, indicators):
                engine.update_indicator(symbol, indicator)
                candidates = engine.on_bar(symbol, bar, day)
                persist_candidates(candidates)
```

### ClickHouse Queries

**Bars:**
```sql
SELECT ts, open, high, low, close, volume, vwap
FROM stock_bar FINAL
WHERE symbol = ? AND period = ? AND session = 'regular'
  AND toDate(toTimezone(ts, 'America/New_York')) = ?
ORDER BY ts
```

**Indicators:**
```sql
SELECT ts, indicator, value
FROM indicator FINAL
WHERE symbol = ? AND timeframe = ? AND session = ?
  AND warm = true
ORDER BY ts
```

**Prior session:**
```sql
SELECT
    argMax(close, ts) AS prior_close,
    max(high)         AS prior_high,
    min(low)          AS prior_low
FROM stock_bar FINAL
WHERE symbol = ? AND period = ? AND session = 'regular'
  AND toDate(toTimezone(ts, 'America/New_York')) = ?
```

### Indicator Merging

Indicators are stored as EAV rows (one row per indicator per bar).
Pivot them per-bar timestamp before passing to the engine:

```python
# Group indicator rows by ts
for ts, rows in groupby(indicator_rows, key=lambda r: r.ts):
    ema = {r.indicator: r.value for r in rows if r.indicator.startswith('ema')}
    atr = next((r.value for r in rows if r.indicator == 'atr'), None)
    engine.update_indicator(symbol, ema_dict=ema, atr_value=atr)
```

### Progress Reporting

```
[ReplayBreakouts] 2021-01-04: NVDA 5m → 2 candidates
[ReplayBreakouts] 2021-01-04: AMD 5m → 0 candidates
...
[ReplayBreakouts] Complete: 1,234 trading days, 45,678 candidates written
[ReplayBreakouts] Elapsed: 142s
```

### Error Handling

- Per-symbol per-day errors: log and continue
- Missing bars for a symbol/day: skip silently
- Missing indicators: skip (symbol not warm for that period on that day)

---

## breakout_candidates Table

Current schema needs a `timeframe` column added:

```sql
ALTER TABLE trading.breakout_candidates
    ADD COLUMN timeframe LowCardinality(String) DEFAULT '5m';
```

Full target schema:
```sql
CREATE TABLE IF NOT EXISTS trading.breakout_candidates (
    symbol          LowCardinality(String),
    timeframe       LowCardinality(String),
    ts              DateTime64(3, 'UTC'),
    session         Date,
    direction       Enum8('long' = 1, 'short' = -1),
    price           Float64,
    level_price     Float64,
    level_age_min   UInt16,
    ribbon_state    LowCardinality(String),
    ribbon_age      UInt16,
    ribbon_spread   Float64,
    atr             Float64,
    bar_range_atr   Float64,
    bar_close_pct   Float64,
    volume_ratio    Float64,
    gap_pct         Float64,
    score           Float32,
    source          Enum8('live' = 1, 'replay' = 2)
) ENGINE = ReplacingMergeTree()
ORDER BY (symbol, timeframe, ts);
```

Note `source` column distinguishes live vs replay rows.

---

## BreakoutConfig Considerations

`SignalConfig` thresholds were designed for 5m bars. For 1m bars these
may be too strict or produce too many candidates. For 60m bars some
thresholds may be too loose.

**Recommendation: use loose thresholds for the replay** — cast a wide
net, let Model 2 learn what matters. The ML model will discover which
candidates across which timeframes and symbols have good outcomes.

The only hard filter worth keeping: `warm = true` on indicators.

---

## Implementation Order

1. Add `timeframe` column to `breakout_candidates` table (ClickHouse DDL)
2. Create `ml/models/breakout/engine.py` — extract `BreakoutEngine`
3. Refactor `detector_service.py` to use `BreakoutEngine` + multi-timeframe
4. Create `tools/replay_breakouts.py` — historical replay
5. Run replay: all symbols, all timeframes, 2020-12-31 → today
6. Verify candidate counts look reasonable per symbol/timeframe/direction

---

## Out of Scope (this phase)

- Outcome labeling (separate script, after replay)
- Options IV feature extraction (separate script, after labeling)
- Model 2 training
- ZMQ publication of signals (phase >= 1)
