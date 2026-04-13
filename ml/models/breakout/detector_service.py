"""
ml/models/breakout/detector_service.py

Multi-timeframe breakout detection service. Consumes bars and EMA/ATR
indicators from MDS via ZMQ subscription across all timeframes, and
generates breakout signals using BreakoutEngine.
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
from ml.shared.mds_client import get_bars, get_prior_session, subscribe_with_backfill

from ml.models.breakout.engine import BreakoutEngine
from ml.models.breakout.persistence import BreakoutPersistor
from ml.models.breakout.signal_generator import BreakoutCandidate, SignalConfig
from ml.models.breakout.types import Bar

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")

TIMEFRAMES = ["1m", "5m", "10m", "15m", "20m", "30m", "60m"]
CAL_TOPIC_PREFIX = "cal."


@dataclass
class BreakoutConfig:
    """Configuration for breakout detection."""
    phase: int = 0                       # 0=log only, 1=publish signals
    level_age_threshold: int = 30        # Min minutes level must be aged
    ribbon_age_threshold: int = 3        # Min bars in ordered state
    ribbon_spread_min_pct: float = 0.1   # Min ribbon spread %
    break_bar_atr_min: float = 0.5       # Min bar range in ATR
    break_bar_close_pct: float = 0.6     # Bar close position threshold
    volume_ratio_min: float = 1.0        # Min volume vs average
    clear_air_atr_min: float = 0.5       # Min distance to prior level
    gap_atr_threshold: float = 2.0       # Max gap in ATR units

    def to_signal_config(self) -> SignalConfig:
        """Convert to SignalConfig for BreakoutEngine."""
        return SignalConfig(
            level_age_threshold=self.level_age_threshold,
            ribbon_age_threshold=self.ribbon_age_threshold,
            ribbon_spread_min_pct=self.ribbon_spread_min_pct,
            break_bar_atr_min=self.break_bar_atr_min,
            break_bar_close_pct=self.break_bar_close_pct,
            volume_ratio_min=self.volume_ratio_min,
            clear_air_atr_min=self.clear_air_atr_min,
            gap_atr_threshold=self.gap_atr_threshold,
        )


class BreakoutDetector:
    """
    Multi-timeframe breakout detection service.

    Runs one BreakoutEngine per timeframe, all within a single process.
    Subscribes to bars and indicators from MDS for all timeframes and
    routes messages to the correct engine.
    """

    def __init__(self, config: BreakoutConfig = None, dry_run: bool = True,
                 timeframes: list[str] = None):
        self.config = config or BreakoutConfig()
        self.dry_run = dry_run
        self.timeframes = timeframes or TIMEFRAMES
        self.symbols: list[str] = []

        # One engine per timeframe
        self.engines: dict[str, BreakoutEngine] = {}

        # Topic prefix -> (timeframe, kind) for O(1) routing
        self._topic_routes: dict[str, tuple[str, str]] = {}

        # MDS connection
        self._mds_router_url: Optional[str] = None
        self._context: Optional[zmq.Context] = None
        self._sub: Optional[zmq.Socket] = None

        # Persistence
        self._persistor: Optional[BreakoutPersistor] = None

        # Control
        self._running = False
        self._today: Optional[date] = None
        self._signals_today: dict[str, list[BreakoutCandidate]] = defaultdict(list)

    def start(self):
        """Start the breakout detector service."""
        log.info("Starting Multi-Timeframe Breakout Detector...")
        log.info(f"Timeframes: {self.timeframes}")
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

        # Create engines
        signal_config = self.config.to_signal_config()
        for tf in self.timeframes:
            engine = BreakoutEngine(tf, signal_config)
            engine.init_symbols(self.symbols)
            self.engines[tf] = engine
        log.info(f"Created {len(self.engines)} engines")

        # Build topic routing table
        for tf in self.timeframes:
            self._topic_routes[f"md.equity.bar.{tf}."] = (tf, "bar")
            self._topic_routes[f"md.equity.indicator.ema.{tf}."] = (tf, "ema")
            self._topic_routes[f"md.equity.indicator.atr.{tf}."] = (tf, "atr")

        # Setup ZMQ FIRST — subscribe to all topics before any RPC calls.
        # Required by subscribe_with_backfill ADR for gap-free indicator seeding.
        self._setup_zmq(market_data_url)

        # Fetch prior session data and seed all engines
        self._fetch_prior_session_data()

        # Warm up levels and indicators per timeframe
        for tf in self.timeframes:
            self._warmup_levels(tf)
            self._warmup_indicators(tf)

        # Init persistence
        if not self.dry_run:
            try:
                from ml.shared.clickhouse import get_ch_client
                ch = get_ch_client()
                self._persistor = BreakoutPersistor(ch)
                log.info("ClickHouse persistor initialized")
            except Exception as e:
                log.warning(f"ClickHouse not available, running without persistence: {e}")

        # Set _today AFTER warmup to prevent reset
        self._today = datetime.now(NY).date()

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

    def _fetch_prior_session_data(self):
        """Fetch prior session data from MDS and seed all engines."""
        log.info("Fetching prior session data...")
        for symbol in self.symbols:
            try:
                session = get_prior_session(self._mds_router_url, symbol)
                if session:
                    for engine in self.engines.values():
                        engine.set_prior_session(
                            symbol, session.close, session.high, session.low
                        )
                    log.debug(f"{symbol} prior: close={session.close:.2f}, "
                              f"high={session.high:.2f}, low={session.low:.2f}")
                else:
                    log.warning(f"{symbol}: no prior session data")
            except Exception as e:
                log.warning(f"Failed to fetch prior session for {symbol}: {e}")

    def _warmup_levels(self, tf: str):
        """Warm up levels and volume from today's bars for one timeframe.

        Uses engine.warmup_bar() to avoid triggering breakout detection
        on historical bars.
        """
        today_et = datetime.now(NY).date()
        engine = self.engines[tf]

        for symbol in self.symbols:
            try:
                bars = get_bars(self._mds_router_url, symbol,
                                bar_date=today_et, period=tf, session=1)

                if not bars:
                    log.debug(f"{symbol} {tf}: no bars for level warmup")
                    continue

                for bar_rec in bars:
                    bar = Bar(
                        ts=bar_rec.ts, open=bar_rec.open, high=bar_rec.high,
                        low=bar_rec.low, close=bar_rec.close,
                        volume=bar_rec.volume, vwap=bar_rec.vwap,
                    )
                    engine.warmup_bar(symbol, bar)

                level_age = engine.levels[symbol].high_age_minutes()
                log.info(f"{symbol} {tf}: warmed from {len(bars)} bars "
                         f"(high_age={level_age}min)")

            except Exception as e:
                log.warning(f"Failed to warmup levels for {symbol} {tf}: {e}")

    def _warmup_indicators(self, tf: str):
        """Seed indicator state from MDS via subscribe_with_backfill for one timeframe.

        ZMQ SUB topics must already be subscribed (done in _setup_zmq)
        before this is called — per subscribe_with_backfill ADR.
        """
        engine = self.engines[tf]

        for symbol in self.symbols:
            try:
                snapshot = subscribe_with_backfill(
                    self._mds_router_url, symbol, "indicators", timeframe=tf
                )
                if snapshot and snapshot.get("ok"):
                    snap = snapshot["snapshot"]
                    ema = snap.get("ema", {})
                    atr_data = snap.get("atr", {})
                    engine.update_indicator(
                        symbol,
                        ema_dict=ema,
                        ribbon_state=ema.get("ribbon_state", "WARMING"),
                        ema_warm=ema.get("warm", False),
                        atr_value=atr_data.get("atr"),
                        atr_warm=atr_data.get("warm", False),
                        bar_index=ema.get("bar_index", 0),
                        seq=snapshot.get("seq", 0),
                    )
                    ind = engine.indicators[symbol]
                    log.info(f"{symbol} {tf} indicators: ribbon={ind.ribbon_state} "
                             f"warm={ind.ema_warm} atr={f'{ind.atr:.4f}' if ind.atr else 'N/A'}")
                else:
                    error = snapshot.get("error", "unknown") if snapshot else "timeout"
                    log.warning(f"{symbol} {tf}: subscribe_with_backfill failed: {error}")
            except Exception as e:
                log.warning(f"Failed to seed indicators for {symbol} {tf}: {e}")

    def _setup_zmq(self, market_data_endpoint: str):
        """Setup ZMQ sockets.

        Subscribes to all topics BEFORE any RPC calls — required by the
        subscribe_with_backfill ADR for gap-free indicator seeding.
        """
        self._context = zmq.Context()
        self._sub = self._context.socket(zmq.SUB)
        self._sub.connect(market_data_endpoint)

        # Subscribe to calendar events
        self._sub.setsockopt_string(zmq.SUBSCRIBE, CAL_TOPIC_PREFIX)

        # Subscribe per timeframe × symbol
        topic_count = 0
        for tf in self.timeframes:
            for symbol in self.symbols:
                self._sub.setsockopt_string(
                    zmq.SUBSCRIBE, f"md.equity.bar.{tf}.{symbol}")
                self._sub.setsockopt_string(
                    zmq.SUBSCRIBE, f"md.equity.indicator.ema.{tf}.{symbol}")
                self._sub.setsockopt_string(
                    zmq.SUBSCRIBE, f"md.equity.indicator.atr.{tf}.{symbol}")
                topic_count += 3

        self._sub.setsockopt(zmq.RCVTIMEO, 100)
        log.info(f"Subscribed to {topic_count} topics "
                 f"({len(self.symbols)} symbols x {len(self.timeframes)} timeframes)")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        log.info(f"Received signal {signum}, shutting down...")
        self._running = False

    def _bar_ingestion_loop(self):
        """Receive and process messages from ZMQ."""
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
        """Route incoming ZMQ message to the correct engine."""
        for prefix, (tf, kind) in self._topic_routes.items():
            if topic.startswith(prefix):
                symbol = topic[len(prefix):]
                if symbol not in self.symbols:
                    return
                if kind == "bar":
                    self._handle_bar(tf, symbol, payload)
                elif kind == "ema":
                    self._handle_ema(tf, symbol, payload)
                elif kind == "atr":
                    self._handle_atr(tf, symbol, payload)
                return

        if topic.startswith(CAL_TOPIC_PREFIX):
            self._handle_calendar(topic, payload)

    def _handle_bar(self, tf: str, symbol: str, payload: str):
        """Process completed bar from MDS."""
        try:
            data = json.loads(payload)
            bar_data = data["data"]

            session = bar_data.get("session", "regular")
            if session != "regular":
                return

            ts = datetime.fromisoformat(
                bar_data["ts"].replace("Z", "+00:00")
            ).astimezone(NY)

            bar = Bar(
                ts=ts,
                open=float(bar_data["open"]),
                high=float(bar_data["high"]),
                low=float(bar_data["low"]),
                close=float(bar_data["close"]),
                volume=int(bar_data["volume"]),
                vwap=float(bar_data.get("vwap", bar_data["close"])),
            )

            log.debug(f"{tf} bar: {symbol} {ts.strftime('%H:%M')} "
                      f"O={bar.open:.2f} H={bar.high:.2f} "
                      f"L={bar.low:.2f} C={bar.close:.2f}")

            candidates = self.engines[tf].on_bar(symbol, bar)
            for c in candidates:
                self._handle_signal(c, tf)

        except Exception as e:
            log.warning(f"Error processing {tf} bar for {symbol}: {e}")

    def _handle_ema(self, tf: str, symbol: str, payload: str):
        """Update EMA ribbon state from MDS."""
        try:
            data = json.loads(payload)
            self.engines[tf].update_indicator(
                symbol,
                ema_dict=data,
                ribbon_state=data.get("ribbon_state"),
                ema_warm=data.get("warm"),
                bar_index=data.get("bar_index"),
                seq=data.get("seq", 0),
            )
        except Exception as e:
            log.warning(f"Error processing EMA for {symbol} {tf}: {e}")

    def _handle_atr(self, tf: str, symbol: str, payload: str):
        """Update ATR state from MDS."""
        try:
            data = json.loads(payload)
            self.engines[tf].update_indicator(
                symbol,
                atr_value=data.get("atr"),
                atr_warm=data.get("warm"),
                seq=data.get("seq", 0),
            )
        except Exception as e:
            log.warning(f"Error processing ATR for {symbol} {tf}: {e}")

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
                if self._persistor:
                    self._persistor.flush()

        except Exception as e:
            log.warning(f"Error processing calendar: {e}")

    def _handle_signal(self, candidate: BreakoutCandidate, timeframe: str):
        """Handle detected breakout signal."""
        log.info(f"BREAKOUT [{timeframe}]: {candidate.symbol} "
                 f"{candidate.direction.value} @ {candidate.price:.2f} "
                 f"(score={candidate.score})")

        self._signals_today[candidate.symbol].append(candidate)

        if self._persistor:
            self._persistor.persist(
                candidate,
                timeframe=timeframe,
                session_date=datetime.now(NY).date(),
                source="live",
            )

    def _reset_session_state(self):
        """Reset state for new trading session."""
        log.info("Resetting session state for all timeframes")
        self._today = datetime.now(NY).date()
        self._signals_today.clear()

        for engine in self.engines.values():
            engine.reset_session(self.symbols)

        self._fetch_prior_session_data()
        for tf in self.timeframes:
            self._warmup_indicators(tf)

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
        if self._persistor:
            self._persistor.flush()
        if self._sub:
            self._sub.close()
        if self._context:
            self._context.term()


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Timeframe Breakout Detector")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry run mode (no persistence)")
    parser.add_argument("--timeframes", type=str, default=None,
                        help="Comma-separated timeframes (default: all)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    timeframes = args.timeframes.split(",") if args.timeframes else None
    config = BreakoutConfig()
    detector = BreakoutDetector(config=config, dry_run=args.dry_run,
                                timeframes=timeframes)
    detector.start()


if __name__ == "__main__":
    main()
