"""
score_session.py

ML session scorer for first-hour directional prediction.

Subscribes to pre-built 1-minute bars from the Scala MDS (md.equity.bar.1m.*)
and calendar events (cal.market.*). At the prediction window (default 10:30am)
computes session features, scores the directional model, and publishes
the prediction back onto the bus.

IMPORTANT: Must be started BEFORE market open (9:30am ET). If started after
market open, the scorer will skip today and wait for the next session.

Advertises itself via UDP broadcast so the Scala layer can discover it.

Service name:  mlScorer
PUB port:      6040  (predictions)
Router port:   6041  (control -- reserved for future use)

Published prediction topic: ml.session.prediction.{SYMBOL}

Prediction message format:
{
  "type": "session_prediction",
  "symbol": "NVDA",
  "timestamp": "2026-03-20T10:30:00-05:00",
  "window_minutes": 60,
  "prediction": "fade_the_high",
  "label": 1,
  "confidence": 0.81,
  "model_version": "nvda_directional_model",
  "features": {
    "w_vwap_dev": -0.0023,
    "sweep_direction": 1,
    "directional_bias": 0.42,
    "target_qqq_corr": 0.71
  }
}

Usage:
    python ml/scoring/score_session.py
    python ml/scoring/score_session.py --symbols NVDA AMD --window 60
    python ml/scoring/score_session.py --symbols NVDA --window 30 --dry-run

Dependencies:
    pip install pyzmq xgboost pandas numpy python-dotenv

Environment variables:
    MARKET_DATA_HOST     ZMQ market data PUB host (default: discovered via UDP)
    MARKET_DATA_PORT     ZMQ market data PUB port (default: 6006)
    CLICKHOUSE_USER, CLICKHOUSE_PASSWORD, CLICKHOUSE_DATABASE
    MODEL_DIR            Directory containing trained model .json files (default: data/)
"""

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import clickhouse_connect
import numpy as np
import pandas as pd
import xgboost as xgb
import zmq
from dotenv import load_dotenv

from discovery import ServiceLocator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NY = ZoneInfo("America/New_York")

SERVICE_NAME   = "mlScorer"
PUB_PORT       = 6040   # Prediction PUB socket (6010/6011 used by IB service)
ROUTER_PORT    = 6041   # Control router (reserved)
DISCOVERY_PORT = 6001   # UDP broadcast port (matches SystemConfig.discovery.udpPort)
BROADCAST_ADDR = "255.255.255.255"
ADVERTISE_INTERVAL = 5  # seconds

PREDICTION_TOPIC_PREFIX = "ml.session.prediction."
BAR_TOPIC_PREFIX        = "md.equity.bar.1m."
CAL_TOPIC_PREFIX        = "cal.market."

# Symbols needed for correlation features
CORRELATION_SYMBOLS = ["QQQ"]   # SMH not in default feed; QQQ sufficient

# History window to pull from ClickHouse for rolling features
HISTORY_SESSIONS = 90

# Features must match train_model.py exactly
FEATURES = [
    "gap_pct", "prior_day_range_pct", "atr20",
    "w_range_pct", "w_range_atr", "w_vwap_dev",
    "f15_range_ratio", "f15_vol_ratio",
    "w_vol_ratio",
    "sweep_signal", "sweep_direction",
    "reversal_progress", "close_position",
    "rolling_reversal_rate", "rolling_high_set_rate",
    "directional_bias", "gap_regime_alignment",
    "target_qqq_corr", "target_smh_corr", "target_qqq_beta",
]

DIRECTIONAL_NAMES = {0: "buy_the_dip", 1: "fade_the_high"}

# Top features to include in published message for Scala layer context
TOP_FEATURES_TO_PUBLISH = [
    "w_vwap_dev", "sweep_direction", "close_position",
    "directional_bias", "target_qqq_corr",
]


# ---------------------------------------------------------------------------
# 1-Minute Bar (received from Scala MDS)
# ---------------------------------------------------------------------------

class Bar:
    """
    Single 1-minute OHLCV bar received from Scala MDS.

    Created from JSON payload:
    {
      "type": "bar_1m",
      "symbol": "NVDA",
      "data": {
        "ts": "2026-03-28T10:30:00-04:00",
        "open": 120.50, "high": 121.00, "low": 120.25, "close": 120.75,
        "volume": 15000, "vwap": 120.62, "trade_count": 142, "session": "regular"
      }
    }
    """
    __slots__ = ["ts", "open", "high", "low", "close", "volume", "vwap", "trade_count", "session"]

    def __init__(self, ts: datetime, open_: float, high: float, low: float,
                 close: float, volume: int, vwap: float, trade_count: int,
                 session: str):
        self.ts          = ts
        self.open        = open_
        self.high        = high
        self.low         = low
        self.close       = close
        self.volume      = volume
        self.vwap        = vwap
        self.trade_count = trade_count
        self.session     = session

    @classmethod
    def from_json(cls, data: dict) -> "Bar":
        """Create Bar from JSON payload."""
        bar_data = data["data"]
        ts_str = bar_data["ts"]
        ts = datetime.fromisoformat(ts_str).astimezone(NY)
        return cls(
            ts          = ts,
            open_       = float(bar_data["open"]),
            high        = float(bar_data["high"]),
            low         = float(bar_data["low"]),
            close       = float(bar_data["close"]),
            volume      = int(bar_data["volume"]),
            vwap        = float(bar_data["vwap"]),
            trade_count = int(bar_data["trade_count"]),
            session     = bar_data.get("session", "regular"),
        )


class BarStore:
    """
    Stores 1-minute bars received from the Scala MDS.
    Thread-safe for concurrent bar ingestion and reading.
    """

    def __init__(self):
        self._bars: dict[str, dict[datetime, Bar]] = defaultdict(dict)
        self._lock = threading.Lock()

    def add_bar(self, symbol: str, bar: Bar):
        """Store a bar received from MDS."""
        with self._lock:
            self._bars[symbol][bar.ts] = bar

    def get_bars(self, symbol: str) -> list[Bar]:
        """Return sorted list of all bars for the symbol."""
        with self._lock:
            return sorted(self._bars[symbol].values(), key=lambda b: b.ts)

    def get_window_bars(self, symbol: str,
                        session_date: date,
                        window_minutes: int) -> list[Bar]:
        """Return bars within the prediction window from open."""
        ro = datetime(session_date.year, session_date.month, session_date.day,
                      9, 30, tzinfo=NY)
        cutoff = ro + timedelta(minutes=window_minutes)
        with self._lock:
            return sorted(
                [b for b in self._bars[symbol].values()
                 if ro <= b.ts < cutoff],
                key=lambda b: b.ts
            )

    def clear(self):
        """Clear all bars (call at start of new session)."""
        with self._lock:
            self._bars.clear()

    def symbol_count(self, symbol: str) -> int:
        """Return count of bars for a symbol."""
        with self._lock:
            return len(self._bars[symbol])


# ---------------------------------------------------------------------------
# Service Advertiser (Python UDP broadcast)
# ---------------------------------------------------------------------------

class ServiceAdvertiser:
    """
    Broadcasts UDP packets to advertise this service on the discovery bus.
    Mirrors the Scala ServiceAdvertiser behavior.
    """

    def __init__(self, service_name: str, pub_port: int, router_port: int,
                 interval: int = ADVERTISE_INTERVAL):
        self._service_name = service_name
        self._pub_port     = pub_port
        self._router_port  = router_port
        self._interval     = interval
        self._running      = False
        self._thread: Optional[threading.Thread] = None
        self._host         = self._get_local_ip()

    @staticmethod
    def _get_local_ip() -> str:
        """Get the local IPv4 address (same logic as Scala NetworkUtils)."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _build_payload(self) -> bytes:
        payload = {
            "service": self._service_name,
            "host":    self._host,
            "pubSub":  f"tcp://{self._host}:{self._pub_port}",
            "router":  f"tcp://{self._host}:{self._router_port}",
            # Also include port for Python ServiceLocator compatibility
            "port":    self._pub_port,
        }
        return json.dumps(payload).encode("utf-8")

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        consecutive_failures = 0
        while self._running:
            try:
                sock.sendto(self._build_payload(),
                            (BROADCAST_ADDR, DISCOVERY_PORT))
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures == 1 or consecutive_failures % 10 == 0:
                    log.warning(f"[ServiceAdvertiser] Broadcast failed "
                                f"(count={consecutive_failures}): {e}")
            time.sleep(self._interval)
        sock.close()

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True,
                                          name="ServiceAdvertiser")
        self._thread.start()
        log.info(f"[ServiceAdvertiser] Broadcasting {self._service_name} "
                 f"at {self._host} pubSub={self._pub_port} router={self._router_port}")

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Historical feature computation
# ---------------------------------------------------------------------------

HISTORY_SQL = """
SELECT
    toDate(ts)                              AS session_date,
    argMax(close, ts)                       AS close_400,
    max(high)                               AS session_high,
    min(low)                                AS session_low,
    sum(volume)                             AS session_volume,
    argMin(open, ts)                        AS open_930
FROM stock_bars_1m
WHERE symbol  = %(symbol)s
  AND session = 1
  AND toDate(ts) < today()
ORDER BY session_date DESC
LIMIT %(n)s
"""

PRIOR_CLOSE_SQL = """
SELECT argMax(close, ts) AS prev_close
FROM stock_bars_1m
WHERE symbol  = %(symbol)s
  AND session = 1
  AND toDate(ts) = %(prev_date)s
"""

QQQ_HISTORY_SQL = """
SELECT
    toDate(ts)          AS session_date,
    argMax(close, ts)   AS close
FROM stock_bars_1m
WHERE symbol  = 'QQQ'
  AND session = 1
  AND toDate(ts) < today()
ORDER BY session_date DESC
LIMIT %(n)s
"""

# Regime features from session_labels table
REGIME_SQL = """
SELECT
    fh_high_is_session_high,
    fh_low_is_session_low,
    session_date
FROM session_labels
WHERE symbol         = %(symbol)s
  AND window_minutes = 60
  AND session_date   < today()
ORDER BY session_date DESC
LIMIT 30
"""


def fetch_history(ch_client, symbol: str,
                  n: int = HISTORY_SESSIONS) -> pd.DataFrame:
    """Fetch last N completed sessions for rolling feature computation."""
    result = ch_client.query(
        HISTORY_SQL,
        parameters={"symbol": symbol, "n": n}
    )
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    df["session_date"] = pd.to_datetime(df["session_date"])
    return df.sort_values("session_date").reset_index(drop=True)


def fetch_prev_close(ch_client, symbol: str,
                     prev_date: date) -> Optional[float]:
    """Fetch the prior session close for gap computation."""
    result = ch_client.query(
        PRIOR_CLOSE_SQL,
        parameters={"symbol": symbol,
                    "prev_date": prev_date.isoformat()}
    )
    rows = result.result_rows
    if rows and rows[0][0] is not None:
        return float(rows[0][0])
    return None


def fetch_qqq_history(ch_client,
                      n: int = HISTORY_SESSIONS) -> pd.Series:
    """Fetch QQQ close history for correlation computation."""
    result = ch_client.query(QQQ_HISTORY_SQL, parameters={"n": n})
    df = pd.DataFrame(result.result_rows, columns=["session_date", "close"])
    df["session_date"] = pd.to_datetime(df["session_date"])
    df = df.sort_values("session_date").reset_index(drop=True)
    return df.set_index("session_date")["close"]


def fetch_regime_features(ch_client, symbol: str) -> dict:
    """
    Fetch rolling regime features from session_labels.
    Returns dict with rolling_reversal_rate, rolling_high_set_rate,
    directional_bias keys.
    Falls back to neutral values if insufficient history.
    """
    result = ch_client.query(
        REGIME_SQL,
        parameters={"symbol": symbol}
    )
    rows = result.result_rows
    if len(rows) < 15:
        # Insufficient history -- use neutral values
        log.warning(f"Insufficient session_labels history for {symbol} "
                    f"({len(rows)} rows) -- using neutral regime features")
        return {
            "rolling_reversal_rate": 0.92,
            "rolling_high_set_rate": 0.50,
            "directional_bias":      0.0,
        }
    fh_high = [r[0] for r in rows]
    fh_low  = [r[1] for r in rows]
    is_reversal = [1 if h or l else 0 for h, l in zip(fh_high, fh_low)]

    rolling_reversal_rate = sum(is_reversal) / len(is_reversal)
    rolling_high_set_rate = sum(fh_high)     / len(fh_high)
    directional_bias      = (rolling_high_set_rate - 0.5) * 2

    return {
        "rolling_reversal_rate": rolling_reversal_rate,
        "rolling_high_set_rate": rolling_high_set_rate,
        "directional_bias":      directional_bias,
    }


# ---------------------------------------------------------------------------
# Feature vector computation
# ---------------------------------------------------------------------------

def bars_to_agg(bars: list[Bar]) -> Optional[dict]:
    """Aggregate a list of Bar objects into OHLCV summary."""
    if not bars:
        return None
    return {
        "high":   max(b.high   for b in bars),
        "low":    min(b.low    for b in bars),
        "close":  bars[-1].close,
        "open":   bars[0].open,
        "volume": sum(b.volume for b in bars),
        "vwap":   (sum(b.close * b.volume for b in bars) /
                   max(sum(b.volume for b in bars), 1)),
    }


def compute_live_features(
    symbol: str,
    today: date,
    window_minutes: int,
    w_bars: list[Bar],
    f15_bars: list[Bar],
    history_df: pd.DataFrame,
    prev_close: float,
    qqq_hist: pd.Series,
    regime: dict,
) -> Optional[pd.Series]:
    """
    Compute a single-row feature vector for today's session.

    Uses:
      - history_df: last N completed sessions from ClickHouse
      - w_bars: window bars from BarStore (pre-extracted)
      - f15_bars: first 15-minute bars from BarStore (pre-extracted)
      - prev_close: yesterday's closing price
      - qqq_hist: QQQ close history for correlation
      - regime: dict with rolling_reversal_rate, rolling_high_set_rate,
                directional_bias from session_labels

    Returns a pd.Series with all FEATURES, or None if insufficient data.
    """
    if not w_bars:
        log.warning(f"No window bars for {symbol} yet")
        return None

    if not f15_bars:
        log.warning(f"No f15 bars for {symbol} yet")
        return None

    w   = bars_to_agg(w_bars)
    f15 = bars_to_agg(f15_bars)

    open_930 = w_bars[0].open
    w_range  = w["high"] - w["low"]
    f15_range = f15["high"] - f15["low"]

    if w_range == 0:
        log.warning(f"Zero window range for {symbol} -- skipping")
        return None

    # Rolling features from history
    hist_ranges = history_df["session_high"] - history_df["session_low"]
    atr20       = hist_ranges.iloc[-20:].mean() if len(hist_ranges) >= 5 else hist_ranges.mean()

    avg_vol_20 = history_df["session_volume"].iloc[-20:].mean() \
                 if len(history_df) >= 5 else history_df["session_volume"].mean()

    # Regime features from session_labels
    rolling_reversal_rate = regime["rolling_reversal_rate"]
    rolling_high_set_rate = regime["rolling_high_set_rate"]
    directional_bias      = regime["directional_bias"]

    gap_pct              = (open_930 - prev_close) / prev_close if prev_close else 0.0
    gap_regime_alignment = gap_pct * directional_bias

    # Sweep signal
    f15_up_ext   = f15["high"] - open_930
    f15_down_ext = open_930 - f15["low"]
    significance = w_range * 0.25

    sweep_up   = (f15_up_ext   > significance) and (w["close"] < open_930)
    sweep_down = (f15_down_ext > significance) and (w["close"] > open_930)
    sweep_signal    = 1 if (sweep_up or sweep_down) else 0
    sweep_direction = 1 if sweep_up else (-1 if sweep_down else 0)

    # Timing features
    high_bar_idx = max(range(len(w_bars)), key=lambda i: w_bars[i].high)
    low_bar_idx  = min(range(len(w_bars)), key=lambda i: w_bars[i].low)
    mins_to_high = high_bar_idx
    mins_to_low  = low_bar_idx

    # Confirmation features
    reversal_progress = (w["high"] - w["close"]) / w_range
    close_position    = (w["close"] - w["low"])  / w_range

    # Correlation features from QQQ history
    target_hist_ret = history_df["close_400"].pct_change().dropna()
    qqq_ret         = qqq_hist.pct_change().dropna()

    # Align on common dates
    common_idx = target_hist_ret.index.intersection(
        pd.to_datetime(qqq_ret.index)
    ) if hasattr(qqq_ret.index, 'intersection') else target_hist_ret.index

    if len(common_idx) >= 20:
        t_ret = target_hist_ret.values[-min(60, len(target_hist_ret)):]
        q_ret = qqq_ret.values[-min(60, len(qqq_ret)):]
        min_len = min(len(t_ret), len(q_ret))
        if min_len >= 20:
            target_qqq_corr = float(np.corrcoef(
                t_ret[-min_len:], q_ret[-min_len:]
            )[0, 1])
            # Beta = cov / var
            cov  = np.cov(t_ret[-min_len:], q_ret[-min_len:])[0, 1]
            var  = np.var(q_ret[-min_len:])
            target_qqq_beta = float(cov / var) if var > 0 else 1.5
        else:
            target_qqq_corr = 0.75
            target_qqq_beta = 1.5
    else:
        target_qqq_corr = 0.75
        target_qqq_beta = 1.5

    # SMH not in default feed -- use QQQ correlation as proxy
    target_smh_corr = target_qqq_corr * 1.05  # SMH historically ~5% higher corr

    feat = {
        "gap_pct":               gap_pct,
        "prior_day_range_pct":   (history_df["session_high"].iloc[-1] -
                                   history_df["session_low"].iloc[-1]) /
                                  prev_close if prev_close else 0.02,
        "atr20":                 atr20,
        "w_range_pct":           w_range / open_930,
        "w_range_atr":           w_range / atr20 if atr20 > 0 else 1.0,
        "w_vwap_dev":            (w["close"] - w["vwap"]) / w["vwap"]
                                  if w["vwap"] > 0 else 0.0,
        "f15_range_ratio":       f15_range / w_range,
        "f15_vol_ratio":         f15["volume"] / max(w["volume"], 1),
        "w_vol_ratio":           w["volume"] / avg_vol_20
                                  if avg_vol_20 > 0 else 0.3,
        "sweep_signal":          sweep_signal,
        "sweep_direction":       sweep_direction,
        "reversal_progress":     reversal_progress,
        "close_position":        close_position,
        "rolling_reversal_rate": rolling_reversal_rate,
        "rolling_high_set_rate": rolling_high_set_rate,
        "directional_bias":      directional_bias,
        "gap_regime_alignment":  gap_regime_alignment,
        "target_qqq_corr":       target_qqq_corr,
        "target_smh_corr":       min(target_smh_corr, 1.0),
        "target_qqq_beta":       target_qqq_beta,
    }

    return pd.Series(feat)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(symbol: str, model_dir: str) -> Optional[xgb.XGBClassifier]:
    """Load the directional model for a symbol."""
    path = Path(model_dir) / f"{symbol.lower()}_directional_model.json"
    if not path.exists():
        # Try w60 variant
        path = Path(model_dir) / f"{symbol.lower()}_w60_directional_model.json"
    if not path.exists():
        log.warning(f"Model not found: {path}")
        return None
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    log.info(f"Loaded model: {path}")
    return model


# ---------------------------------------------------------------------------
# Scoring server
# ---------------------------------------------------------------------------

class SessionScorer:
    """
    Main scoring server.

    Lifecycle:
      - MUST start BEFORE market open (9:30am ET)
      - Subscribes to 1-minute bars (md.equity.bar.1m.*) from Scala MDS
      - Subscribes to calendar events (cal.market.*) for open/close signals
      - Stores bars in memory as they arrive
      - At prediction_time (10:30am NY): scores each configured symbol
      - Publishes predictions on ZMQ PUB socket
      - Advertises itself via UDP for service discovery

    If started after market open, skips today's session and waits for tomorrow.
    """

    def __init__(
        self,
        symbols: list[str],
        window_minutes: int,
        model_dir: str,
        dry_run: bool = False,
    ):
        self.symbols        = symbols
        self.window_minutes = window_minutes
        self.model_dir      = model_dir
        self.dry_run        = dry_run

        self.bar_store      = BarStore()
        self.models         = {}
        self._scored_today  = set()
        self._running       = False

        # Market state from calendar events
        self._market_open      = False
        self._market_open_time: Optional[datetime] = None
        self._skip_today       = False

        # ZMQ context and sockets (initialized in start())
        self._context:  Optional[zmq.Context] = None
        self._sub:      Optional[zmq.Socket]  = None  # subscribe to market data
        self._pub:      Optional[zmq.Socket]  = None  # publish predictions
        self._advertiser: Optional[ServiceAdvertiser] = None

        # ClickHouse client (lazy-initialized)
        self._ch_client = None

    def _get_ch_client(self):
        if self._ch_client is None:
            log.info("Discovering ClickHouse...")
            ch_endpoint = ServiceLocator.wait_for_service(
                ServiceLocator.CLICKHOUSE, timeout_sec=60
            )
            self._ch_client = clickhouse_connect.get_client(
                host     = ch_endpoint.host,
                port     = ch_endpoint.port,
                username = os.environ.get("CLICKHOUSE_USER",     "default"),
                password = os.environ.get("CLICKHOUSE_PASSWORD", "Aector99"),
                database = os.environ.get("CLICKHOUSE_DATABASE", "trading"),
            )
        return self._ch_client

    def _setup_zmq(self, market_data_endpoint: str):
        self._context = zmq.Context()

        # SUB socket -- subscribe to 1-minute bars and calendar events
        self._sub = self._context.socket(zmq.SUB)
        self._sub.connect(market_data_endpoint)

        # Subscribe to calendar events for market open/close
        self._sub.setsockopt_string(zmq.SUBSCRIBE, CAL_TOPIC_PREFIX)
        log.info(f"Subscribed to {CAL_TOPIC_PREFIX}*")

        # Subscribe to bar topics for all configured symbols
        all_symbols = list(set(self.symbols + CORRELATION_SYMBOLS))
        for sym in all_symbols:
            topic = f"{BAR_TOPIC_PREFIX}{sym}"
            self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)
            log.info(f"Subscribed to {topic}")

        self._sub.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout

        # PUB socket -- publish predictions
        if not self.dry_run:
            self._pub = self._context.socket(zmq.PUB)
            self._pub.bind(f"tcp://*:{PUB_PORT}")
            log.info(f"Prediction PUB socket bound on port {PUB_PORT}")

    def _publish_prediction(self, symbol: str, prediction: dict):
        topic   = f"{PREDICTION_TOPIC_PREFIX}{symbol}"
        payload = json.dumps(prediction)
        if self.dry_run:
            log.info(f"[DRY RUN] Would publish to {topic}: {payload}")
        else:
            self._pub.send_string(f"{topic} {payload}")
            log.info(f"Published prediction for {symbol}: "
                     f"{prediction['prediction']} "
                     f"(confidence={prediction['confidence']:.3f})")

    def _score_symbol(self, symbol: str, today: date) -> Optional[dict]:
        """Compute features and score for one symbol."""
        model = self.models.get(symbol)
        if model is None:
            log.warning(f"No model loaded for {symbol}")
            return None

        ch = self._get_ch_client()

        # Fetch history and prior close
        history_df = fetch_history(ch, symbol)
        if len(history_df) < 25:
            log.warning(f"Insufficient history for {symbol}: {len(history_df)} sessions")
            return None

        # Prior session date -- last date in history
        prev_date  = history_df["session_date"].iloc[-1].date()
        prev_close = fetch_prev_close(ch, symbol, prev_date)
        if prev_close is None:
            log.warning(f"Could not fetch prior close for {symbol}")
            return None

        # QQQ history for correlation
        qqq_hist = fetch_qqq_history(ch)

        # Regime features from session_labels
        regime = fetch_regime_features(ch, symbol)

        # Get bars from store
        w_bars = self.bar_store.get_window_bars(symbol, today, self.window_minutes)
        f15_bars = self.bar_store.get_window_bars(symbol, today, 15)

        # Compute feature vector
        features = compute_live_features(
            symbol         = symbol,
            today          = today,
            window_minutes = self.window_minutes,
            w_bars         = w_bars,
            f15_bars       = f15_bars,
            history_df     = history_df,
            prev_close     = prev_close,
            qqq_hist       = qqq_hist,
            regime         = regime,
        )

        if features is None:
            return None

        # Score
        X         = pd.DataFrame([features[FEATURES]])
        probs     = model.predict_proba(X)[0]
        label     = int(probs.argmax())
        confidence = float(probs.max())

        # Build message
        now = datetime.now(NY)
        prediction = {
            "type":          "session_prediction",
            "symbol":        symbol,
            "timestamp":     now.isoformat(),
            "window_minutes": self.window_minutes,
            "prediction":    DIRECTIONAL_NAMES[label],
            "label":         label,
            "confidence":    round(confidence, 4),
            "model_version": f"{symbol.lower()}_directional_model",
            "features":      {k: round(float(features[k]), 6)
                               for k in TOP_FEATURES_TO_PUBLISH
                               if k in features},
        }

        return prediction

    def _is_prediction_time(self, now: datetime) -> bool:
        """Check if it's time to score (within 30s after prediction window close)."""
        target_hour   = 9 + self.window_minutes // 60
        target_minute = 30 + self.window_minutes % 60
        if target_minute >= 60:
            target_hour += 1
            target_minute -= 60
        return (
            now.hour == target_hour and
            now.minute == target_minute and
            now.second < 30
        )

    def _is_market_hours(self, now: datetime) -> bool:
        """Check if we're in or near regular session hours."""
        return (
            now.weekday() < 5 and  # Mon-Fri
            (now.hour > 9 or (now.hour == 9 and now.minute >= 25)) and
            now.hour < 16
        )

    def start(self):
        """Start the scoring server."""
        self._running = True

        # Load models
        for sym in self.symbols:
            model = load_model(sym, self.model_dir)
            if model:
                self.models[sym] = model
            else:
                log.warning(f"No model for {sym} -- will skip scoring")

        if not self.models and not self.dry_run:
            log.error("No models loaded -- exiting")
            return

        # Discover market data service
        log.info("Discovering market data service...")
        md_endpoint = ServiceLocator.wait_for_service(
            ServiceLocator.MARKET_DATA, timeout_sec=60
        )
        market_data_url = md_endpoint.pubSub if hasattr(md_endpoint, 'pubSub') \
                          else f"tcp://{md_endpoint.host}:6006"

        self._setup_zmq(market_data_url)

        # Start service advertiser
        self._advertiser = ServiceAdvertiser(
            service_name = SERVICE_NAME,
            pub_port     = PUB_PORT,
            router_port  = ROUTER_PORT,
        )
        self._advertiser.start()

        log.info(f"Session scorer started. Symbols: {self.symbols}  "
                 f"Window: {self.window_minutes}min  "
                 f"{'DRY RUN' if self.dry_run else 'LIVE'}")

        self._run_loop()

    def _run_loop(self):
        """Main event loop."""
        today = date.today()

        while self._running:
            now = datetime.now(NY)

            # Reset state at start of new day
            if now.date() != today:
                today = now.date()
                self._scored_today.clear()
                self.bar_store.clear()
                self._skip_today = False
                self._market_open = False
                self._market_open_time = None
                log.info(f"New session: {today}")

            # Only process during market hours
            if not self._is_market_hours(now):
                time.sleep(5)
                continue

            # Late-start check: if we haven't seen market open but it's past 10am,
            # we missed the open event -- skip today
            if (self._market_open_time is None and
                not self._skip_today and
                now.hour >= 10):
                log.warning("Started after market open -- skipping today's predictions, "
                            "will score from tomorrow")
                self._skip_today = True

            # Ingest incoming bar/calendar messages
            try:
                frames = self._sub.recv_multipart()
                if len(frames) >= 2:
                    topic = frames[0].decode("utf-8")
                    payload = frames[1].decode("utf-8")
                    self._handle_message(topic, payload)
            except zmq.Again:
                pass  # No message -- check for scoring time

            # Skip scoring if we missed market open
            if self._skip_today:
                time.sleep(1)
                continue

            # Check if it's time to score
            if self._is_prediction_time(now):
                unsettled = [s for s in self.symbols
                             if s not in self._scored_today]
                for symbol in unsettled:
                    log.info(f"Scoring {symbol} at window={self.window_minutes}min")
                    try:
                        prediction = self._score_symbol(symbol, today)
                        if prediction:
                            self._publish_prediction(symbol, prediction)
                            self._scored_today.add(symbol)
                    except Exception as e:
                        log.error(f"Scoring failed for {symbol}: {e}",
                                  exc_info=True)

            # If all symbols scored, idle until end of day
            if len(self._scored_today) == len(self.symbols):
                log.info("All symbols scored. Idling until market close.")
                time.sleep(30)

    def _handle_message(self, topic: str, payload: str):
        """Route message to appropriate handler based on topic."""
        if topic.startswith(BAR_TOPIC_PREFIX):
            self._handle_bar_message(topic, payload)
        elif topic.startswith(CAL_TOPIC_PREFIX):
            self._handle_calendar_event(topic, payload)

    def _handle_bar_message(self, topic: str, payload: str):
        """Parse and store a 1-minute bar from the Scala MDS."""
        try:
            data = json.loads(payload)
            if data.get("type") != "bar_1m":
                return

            symbol = data["symbol"]
            bar = Bar.from_json(data)
            self.bar_store.add_bar(symbol, bar)

            # Log first bar of the day for each symbol
            if self.bar_store.symbol_count(symbol) == 1:
                log.info(f"First bar received for {symbol}: {bar.ts}")

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            log.debug(f"Bar parse error: {e} -- payload={payload[:80]}")

    def _handle_calendar_event(self, topic: str, payload: str):
        """Handle calendar events (market open/close) from MDS."""
        try:
            data = json.loads(payload)
            event_type = data.get("type", "")

            if topic == "cal.market.open" or event_type == "open":
                self._market_open = True
                self._market_open_time = datetime.now(NY)
                log.info("Market OPEN event received -- beginning bar accumulation")

            elif topic == "cal.market.close" or event_type == "close":
                self._market_open = False
                log.info("Market CLOSE event received")

        except (json.JSONDecodeError, KeyError) as e:
            log.debug(f"Calendar event parse error: {e}")

    def stop(self):
        """Graceful shutdown."""
        self._running = False
        if self._advertiser:
            self._advertiser.stop()
        if self._sub:
            self._sub.close(linger=0)
        if self._pub:
            self._pub.close(linger=0)
        if self._context:
            self._context.term()
        if self._ch_client:
            self._ch_client.close()
        log.info("Session scorer stopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ML session scoring server")
    parser.add_argument("--symbols",   nargs="+", default=["NVDA", "AMD"],
                        help="Symbols to score (default: NVDA AMD)")
    parser.add_argument("--window",    type=int, default=60,
                        help="Prediction window minutes (default: 60)")
    parser.add_argument("--model-dir", default="data",
                        help="Directory containing model .json files (default: data/)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Compute and log predictions without publishing")
    args = parser.parse_args()

    scorer = SessionScorer(
        symbols        = [s.upper() for s in args.symbols],
        window_minutes = args.window,
        model_dir      = args.model_dir,
        dry_run        = args.dry_run,
    )

    import signal
    def _shutdown(sig, frame):
        log.info("Shutdown signal received")
        scorer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    scorer.start()


if __name__ == "__main__":
    main()
