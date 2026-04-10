"""
ml/models/breakout/detector_service.py

Breakout detection service that consumes 1m bars from MDS,
aggregates to 5m, and generates breakout signals.
"""

import json
import logging
import signal
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

import zmq

from discovery.service_locator import ServiceLocator
from ml.shared.config import fetch_symbol_list
from ml.shared.mds_client import get_bars, get_prior_session

from ml.models.breakout.bar_aggregator import Bar5m, Bar5mAggregator
from ml.models.breakout.ema_ribbon import EMARibbon
from ml.models.breakout.atr_calculator import ATRCalculator, VolumeAverageCalculator
from ml.models.breakout.level_tracker import LevelTracker
from ml.models.breakout.signal_generator import (
    BreakoutConditionChecker, SignalConfig, BreakoutCandidate
)

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")

BAR_5M_TOPIC_PREFIX  = "md.equity.bar.5m."
IND_EMA_TOPIC_PREFIX = "md.equity.indicator.ema.5m."
IND_ATR_TOPIC_PREFIX = "md.equity.indicator.atr.5m."
CAL_TOPIC_PREFIX     = "cal."


@dataclass
class BreakoutConfig:
    """Configuration for breakout detection."""
    phase: int = 0                       # 0=log only, 1=publish signals
    timeframe_minutes: int = 5
    level_age_threshold: int = 30        # Min minutes level must be aged
    ribbon_age_threshold: int = 3        # Min bars in ordered state
    ribbon_spread_min_pct: float = 0.1   # Min ribbon spread %
    break_bar_atr_min: float = 0.5       # Min bar range in ATR
    break_bar_close_pct: float = 0.6     # Bar close position threshold
    volume_ratio_min: float = 1.0        # Min volume vs average
    clear_air_atr_min: float = 0.5       # Min distance to prior level
    gap_atr_threshold: float = 2.0       # Max gap in ATR units


@dataclass
class IndicatorState:
    """Latest indicator values from MDS for one symbol."""
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


class BreakoutDetector:
    """
    Main breakout detection service.

    Subscribes to 1m bars from MDS, aggregates to 5m bars,
    and checks for breakout conditions on each completed 5m bar.
    """

    def __init__(self, config: BreakoutConfig = None, dry_run: bool = True):
        self.config = config or BreakoutConfig()
        self.dry_run = dry_run
        self.symbols: list[str] = []

        # Per-symbol state
        self.aggregators: dict[str, Bar5mAggregator] = {}
        self.ribbons: dict[str, EMARibbon] = {}
        self.levels: dict[str, LevelTracker] = {}
        self.atr_calcs: dict[str, ATRCalculator] = {}
        self.volume_calcs: dict[str, VolumeAverageCalculator] = {}

        # Session state
        self.prior_close: dict[str, Optional[float]] = defaultdict(lambda: None)
        self.prior_session_high: dict[str, Optional[float]] = defaultdict(lambda: None)
        self.prior_session_low: dict[str, Optional[float]] = defaultdict(lambda: None)
        self.session_open: dict[str, Optional[float]] = defaultdict(lambda: None)
        self.gap_pct: dict[str, float] = defaultdict(float)

        # MDS connection
        self._mds_router_url: Optional[str] = None
        self._context: Optional[zmq.Context] = None
        self._sub: Optional[zmq.Socket] = None

        # Control
        self._running = False
        self._today: Optional[date] = None
        self._signals_today: dict[str, list[BreakoutCandidate]] = defaultdict(list)

    def start(self):
        """Start the breakout detector service."""
        log.info("Starting Breakout Detector...")
        log.info(f"Phase: {self.config.phase}, Dry run: {self.dry_run}")

        # Discover MDS
        log.info("Discovering Market Data Service...")
        md_endpoint = ServiceLocator.wait_for_service(
            ServiceLocator.MARKET_DATA,
            timeout_sec=60,
        )
        market_data_url = md_endpoint.pub_sub
        self._mds_router_url = md_endpoint.router
        log.info(f"MDS found: PUB={market_data_url}, ROUTER={self._mds_router_url}")

        # Fetch trading universe
        log.info("Fetching trading universe...")
        self.symbols = fetch_symbol_list(self._mds_router_url, "trading_universe")
        log.info(f"Trading universe: {len(self.symbols)} symbols")

        # Initialize per-symbol state
        self._init_symbol_state()

        # Fetch prior session data
        self._fetch_prior_session_data()

        # Warmup from today's bars
        log.info("Warming up indicators from MDS...")
        self._warmup_from_history()

        # Set _today AFTER warmup to prevent reset
        self._today = datetime.now(NY).date()

        # Setup ZMQ
        self._setup_zmq(market_data_url)

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start bar ingestion thread
        self._running = True
        ingestion_thread = threading.Thread(
            target=self._bar_ingestion_loop,
            daemon=True,
            name="bar-ingestion",
        )
        ingestion_thread.start()

        # Main loop
        log.info("Breakout Detector started")
        self._main_loop()

    def _init_symbol_state(self):
        """Initialize per-symbol state objects."""
        for symbol in self.symbols:
            self.aggregators[symbol] = Bar5mAggregator(symbol, self.config.timeframe_minutes)
            self.ribbons[symbol] = EMARibbon()
            self.levels[symbol] = LevelTracker(symbol, self.config.timeframe_minutes)
            self.atr_calcs[symbol] = ATRCalculator()
            self.volume_calcs[symbol] = VolumeAverageCalculator()

    def _fetch_prior_session_data(self):
        """Fetch prior session data from MDS for all symbols."""
        log.info("Fetching prior session data...")
        for symbol in self.symbols:
            try:
                session = get_prior_session(self._mds_router_url, symbol)
                if session:
                    self.prior_close[symbol] = session.close
                    self.prior_session_high[symbol] = session.high
                    self.prior_session_low[symbol] = session.low
                    log.debug(f"{symbol} prior: close={session.close:.2f}, "
                              f"high={session.high:.2f}, low={session.low:.2f}")
                else:
                    log.warning(f"{symbol}: no prior session data")
            except Exception as e:
                log.warning(f"Failed to fetch prior session for {symbol}: {e}")

    def _warmup_from_history(self):
        """Warm up indicators from today's bars fetched from MDS."""
        # Use ET date explicitly
        today_et = datetime.now(NY).date()

        for symbol in self.symbols:
            try:
                # Fetch today's bars from MDS
                bars_1m = get_bars(self._mds_router_url, symbol, bar_date=today_et, session=1)

                if not bars_1m:
                    log.debug(f"{symbol}: no MDS bars for warmup")
                    continue

                first_ts = bars_1m[0].ts.strftime("%H:%M")
                last_ts = bars_1m[-1].ts.strftime("%H:%M")
                log.info(f"{symbol}: fetched {len(bars_1m)} 1m bars ({first_ts}-{last_ts})")

                # Re-initialize level tracker with prior session data
                self.levels[symbol] = LevelTracker(
                    symbol,
                    timeframe_minutes=self.config.timeframe_minutes,
                    prior_close=self.prior_close[symbol],
                    prior_session_high=self.prior_session_high[symbol],
                    prior_session_low=self.prior_session_low[symbol],
                )

                # Track session open from first bar
                if self.session_open[symbol] is None:
                    self.session_open[symbol] = bars_1m[0].open
                    if self.prior_close[symbol] is not None:
                        self.gap_pct[symbol] = (
                            (bars_1m[0].open - self.prior_close[symbol])
                            / self.prior_close[symbol]
                        )

                # Feed bars through aggregator
                bar5m_count = 0
                for bar in bars_1m:
                    bar5m = self.aggregators[symbol].add_bar(bar)
                    if bar5m is not None:
                        self._update_indicators(symbol, bar5m)
                        bar5m_count += 1

                if bar5m_count == 0:
                    log.debug(f"{symbol}: no complete 5m bars for warmup")
                    continue

                ribbon = self.ribbons[symbol]
                atr_val = self.atr_calcs[symbol].value or 0.0
                log.info(f"{symbol}: warmed up with {bar5m_count} 5m bars "
                         f"(ribbon={ribbon.state.value}, is_warm={ribbon.is_warm}, "
                         f"high_age={self.levels[symbol].high_age_minutes()}min, "
                         f"atr={atr_val:.2f})")

            except Exception as e:
                log.warning(f"Failed to warmup {symbol}: {e}")

    def _setup_zmq(self, market_data_endpoint: str):
        """Setup ZMQ sockets."""
        self._context = zmq.Context()
        self._sub = self._context.socket(zmq.SUB)
        self._sub.connect(market_data_endpoint)

        # Subscribe to calendar events
        self._sub.setsockopt_string(zmq.SUBSCRIBE, CAL_TOPIC_PREFIX)

        # Subscribe to bars for all symbols
        for symbol in self.symbols:
            topic = f"{BAR_TOPIC_PREFIX}{symbol}"
            self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)

        self._sub.setsockopt(zmq.RCVTIMEO, 100)
        log.info(f"Subscribed to {len(self.symbols)} symbols")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        log.info(f"Received signal {signum}, shutting down...")
        self._running = False

    def _bar_ingestion_loop(self):
        """Receive and process bars from ZMQ."""
        log.info("Bar ingestion thread started")
        while self._running:
            try:
                frames = self._sub.recv_multipart()
                if len(frames) >= 2:
                    topic = frames[0].decode("utf-8")
                    payload = frames[1].decode("utf-8")
                    self._handle_message(topic, payload)
            except zmq.Again:
                pass
            except Exception as e:
                if self._running:
                    log.warning(f"Bar ingestion error: {e}")
        log.info("Bar ingestion thread stopped")

    def _handle_message(self, topic: str, payload: str):
        """Route incoming ZMQ message."""
        if topic.startswith(BAR_TOPIC_PREFIX):
            symbol = topic[len(BAR_TOPIC_PREFIX):]
            if symbol in self.symbols:
                self._handle_bar(symbol, payload)
        elif topic.startswith(CAL_TOPIC_PREFIX):
            self._handle_calendar(topic, payload)

    def _handle_bar(self, symbol: str, payload: str):
        """Process incoming 1-minute bar."""
        try:
            data = json.loads(payload)
            bar_data = data["data"]

            # Parse timestamp (comes in ISO format with offset)
            raw_ts = bar_data["ts"]
            ts = datetime.fromisoformat(raw_ts).astimezone(NY)

            # Skip non-regular session
            session = bar_data.get("session", "regular")
            if session != "regular":
                return

            # Create bar object
            class Bar:
                pass
            bar = Bar()
            bar.ts = ts
            bar.open = float(bar_data["open"])
            bar.high = float(bar_data["high"])
            bar.low = float(bar_data["low"])
            bar.close = float(bar_data["close"])
            bar.volume = int(bar_data["volume"])
            bar.vwap = float(bar_data["vwap"])

            log.debug(f"1m bar: {symbol} {ts.strftime('%H:%M')} C={bar.close:.2f}")

            # Track session open
            if self.session_open[symbol] is None:
                self.session_open[symbol] = bar.open
                if self.prior_close[symbol] is not None:
                    self.gap_pct[symbol] = (
                        (bar.open - self.prior_close[symbol])
                        / self.prior_close[symbol]
                    )
                    log.info(f"{symbol} gap: {self.gap_pct[symbol]*100:.2f}%")

            # Aggregate to 5m
            bar5m = self.aggregators[symbol].add_bar(bar)
            if bar5m is not None:
                log.debug(f"5m bar: {symbol} {bar5m.ts.strftime('%H:%M')} "
                          f"O={bar5m.open:.2f} H={bar5m.high:.2f} "
                          f"L={bar5m.low:.2f} C={bar5m.close:.2f}")
                self._on_bar_5m(symbol, bar5m)

        except Exception as e:
            log.warning(f"Error processing bar for {symbol}: {e}")

    def _handle_calendar(self, topic: str, payload: str):
        """Process calendar event."""
        try:
            data = json.loads(payload)
            event_type = data.get("type", "")

            if event_type == "market_open":
                log.info("Market open event")
                self._reset_session_state()
            elif event_type == "market_close":
                log.info("Market close event")

        except Exception as e:
            log.warning(f"Error processing calendar: {e}")

    def _reset_session_state(self):
        """Reset state for new trading session."""
        log.info("Resetting session state")
        self._today = datetime.now(NY).date()
        self._signals_today.clear()

        for symbol in self.symbols:
            self.aggregators[symbol].clear()
            self.ribbons[symbol].reset()
            self.levels[symbol].clear()
            self.atr_calcs[symbol].reset()
            self.volume_calcs[symbol].reset()
            self.session_open[symbol] = None
            self.gap_pct[symbol] = 0.0

        # Refresh prior session data
        self._fetch_prior_session_data()

    def _on_bar_5m(self, symbol: str, bar5m: Bar5m):
        """Process completed 5-minute bar."""
        self._update_indicators(symbol, bar5m)
        self._check_breakout(symbol, bar5m)

    def _update_indicators(self, symbol: str, bar5m: Bar5m):
        """Update all indicators with new bar."""
        self.ribbons[symbol].update(bar5m)
        self.levels[symbol].update(bar5m)
        self.atr_calcs[symbol].update(bar5m)
        self.volume_calcs[symbol].update(bar5m.volume)

    def _check_breakout(self, symbol: str, bar5m: Bar5m):
        """Check for breakout conditions."""
        ribbon = self.ribbons[symbol]
        levels = self.levels[symbol]
        atr_calc = self.atr_calcs[symbol]
        vol_calc = self.volume_calcs[symbol]

        # Need warmup
        if not ribbon.is_warm or not atr_calc.is_warm:
            log.debug(f"{symbol} not warm: ribbon={ribbon.is_warm} atr={atr_calc.is_warm}")
            return

        atr = atr_calc.value
        avg_vol = vol_calc.value
        volume_ratio = bar5m.volume / avg_vol if avg_vol > 0 else 0

        # Build checker
        signal_config = SignalConfig(
            level_age_threshold=self.config.level_age_threshold,
            ribbon_age_threshold=self.config.ribbon_age_threshold,
            ribbon_spread_min_pct=self.config.ribbon_spread_min_pct,
            break_bar_atr_min=self.config.break_bar_atr_min,
            break_bar_close_pct=self.config.break_bar_close_pct,
            volume_ratio_min=self.config.volume_ratio_min,
            clear_air_atr_min=self.config.clear_air_atr_min,
            gap_atr_threshold=self.config.gap_atr_threshold,
        )
        checker = BreakoutConditionChecker(signal_config)

        # Check long breakout
        if levels.high_price is not None:
            candidate = checker.check_long_breakout(
                symbol=symbol,
                bar5m=bar5m,
                level_price=levels.high_price,
                level_age_minutes=levels.high_age_minutes(),
                ribbon_state=ribbon.state,
                ribbon_age=ribbon.state_age,
                ribbon_spread_pct=ribbon.spread_pct(bar5m.close),
                atr=atr,
                volume_ratio=volume_ratio,
                gap_pct=self.gap_pct[symbol],
                prior_session_high=self.prior_session_high[symbol],
            )
            if candidate:
                self._handle_signal(candidate)

        # Check short breakout
        if levels.low_price is not None:
            candidate = checker.check_short_breakout(
                symbol=symbol,
                bar5m=bar5m,
                level_price=levels.low_price,
                level_age_minutes=levels.low_age_minutes(),
                ribbon_state=ribbon.state,
                ribbon_age=ribbon.state_age,
                ribbon_spread_pct=ribbon.spread_pct(bar5m.close),
                atr=atr,
                volume_ratio=volume_ratio,
                gap_pct=self.gap_pct[symbol],
                prior_session_low=self.prior_session_low[symbol],
            )
            if candidate:
                self._handle_signal(candidate)

    def _handle_signal(self, candidate: BreakoutCandidate):
        """Handle detected breakout signal."""
        log.info(f"BREAKOUT: {candidate.symbol} {candidate.direction.value} "
                 f"@ {candidate.price:.2f} (score={candidate.score})")

        self._signals_today[candidate.symbol].append(candidate)

        # TODO: Persist to ClickHouse
        # TODO: Publish to ZMQ if phase >= 1

    def _main_loop(self):
        """Main loop - monitors session state."""
        while self._running:
            now = datetime.now(NY)

            # Check for new day
            if self._today != now.date():
                self._reset_session_state()

            time.sleep(1)

        # Cleanup
        log.info("Shutting down...")
        if self._sub:
            self._sub.close()
        if self._context:
            self._context.term()


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Breakout Detector Service")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry run mode (no persistence)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = BreakoutConfig()
    detector = BreakoutDetector(config=config, dry_run=args.dry_run)
    detector.start()


if __name__ == "__main__":
    main()
