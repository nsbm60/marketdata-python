"""
score_session.py

ML session scorer for first-hour directional prediction.

Subscribes to pre-built 1-minute bars from the Scala MDS (md.equity.bar.1m.*)
and calendar events (cal.market.*). Uses per-symbol prediction windows from
ml/config/prediction_windows.yaml. For symbols with early-fire windows (e.g., w15),
attempts scoring at the primary window with a confidence threshold; if below
threshold, falls back to scoring at w60. Publishes predictions back onto the bus.

NOTE: Can be started after market open as long as the prediction window hasn't
passed. Health check backfills any missing bars from Alpaca. If started after
the latest prediction window (w60 = 10:30am ET), skips today and waits for
tomorrow.

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
    python ml/scoring/score_session.py --dry-run

Symbols are fetched automatically from MDS "trading_universe" list at startup.

Dependencies:
    pip install pyzmq xgboost pandas numpy python-dotenv

Environment variables:
    MARKET_DATA_HOST     ZMQ market data PUB host (default: discovered via UDP)
    MARKET_DATA_PORT     ZMQ market data PUB port (default: 6006)
    CLICKHOUSE_USER, CLICKHOUSE_PASSWORD, CLICKHOUSE_DATABASE
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
from ml.shared.config import load_prediction_config, PredictionWindowConfig, fetch_symbol_list

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

# Health check settings
HEALTH_CHECK_LOOKBACK_DAYS = 30   # How far back to check
MIN_BARS_PER_SESSION       = 200  # Minimum bars for a complete session
                                  # (accounts for early closes, halts)


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
FROM stock_bar FINAL
WHERE symbol  = %(symbol)s
  AND period  = '1m'
  AND session = 1
  AND toDate(toTimezone(ts, 'America/New_York')) < today()
GROUP BY session_date
ORDER BY session_date DESC
LIMIT %(n)s
"""

PRIOR_CLOSE_SQL = """
SELECT argMax(close, ts) AS prev_close
FROM stock_bar FINAL
WHERE symbol  = %(symbol)s
  AND period  = '1m'
  AND session = 1
  AND toDate(toTimezone(ts, 'America/New_York')) = %(prev_date)s
"""

QQQ_HISTORY_SQL = """
SELECT
    toDate(toTimezone(ts, 'America/New_York'))  AS session_date,
    argMax(close, ts)   AS close
FROM stock_bar FINAL
WHERE symbol  = 'QQQ'
  AND period  = '1m'
  AND session = 1
  AND toDate(toTimezone(ts, 'America/New_York')) < today()
GROUP BY session_date
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

def load_model(symbol: str, window: int = 60) -> Optional[xgb.XGBClassifier]:
    """
    Load the directional model for a symbol using new path structure.

    Looks for model at: data/models/{SYMBOL}/session_direction_w{window}.json
    """
    from ml.shared.paths import model_path

    path = model_path(symbol, "session_direction", window)
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
        dry_run: bool = False,
    ):
        self.symbols: list[str] = []  # Populated in start() from MDS
        self.dry_run = dry_run

        # Load per-symbol prediction window config
        self.pred_config: PredictionWindowConfig = load_prediction_config()
        log.info(f"Loaded prediction config with {len(self.pred_config.symbols)} symbols")

        self.bar_store = BarStore()
        # Models keyed by (symbol, window) tuple
        self.models: dict[tuple[str, int], xgb.XGBClassifier] = {}
        self._scored_today: set[str] = set()
        # Symbols that attempted early-fire but didn't meet confidence threshold
        self._early_attempted: set[str] = set()
        self._idle_logged = False
        self._eod_done = False
        self._running = False

        # Prediction log for outcome tracking
        self._prediction_log: dict[str, dict] = {}  # symbol -> prediction dict

        # Market state from calendar events
        self._market_open = False
        self._market_open_time: Optional[datetime] = None
        self._skip_today = False

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

    # -----------------------------------------------------------------------
    # Startup Health Check
    # -----------------------------------------------------------------------

    def _startup_health_check(self, ch_client) -> None:
        """
        Verify session labels are complete for all symbols.

        Bar backfill is now handled by MDS at startup (BarGapDetector).
        This method only checks and backfills session_labels.

        Checks from HEALTH_CHECK_LOOKBACK_DAYS ago through today.
        """
        today      = date.today()
        start_date = today - timedelta(days=HEALTH_CHECK_LOOKBACK_DAYS)

        # Get trading calendar from SPY session_labels
        trading_days = self._get_trading_days(ch_client, start_date, today)
        if not trading_days:
            log.warning("Could not determine trading calendar -- skipping health check")
            return

        log.info(f"Health check: {len(trading_days)} trading days from "
                 f"{start_date} to {today}")

        # Check bar data status (MDS handles backfill, we just log)
        all_symbols = list(set(self.symbols + CORRELATION_SYMBOLS))
        for symbol in all_symbols:
            gaps = self._find_gaps(ch_client, symbol, trading_days)
            if gaps:
                # MDS should have filled these - log for visibility
                log.warning(f"{symbol}: {len(gaps)} gap days (MDS should have filled): "
                           f"{gaps[:5]}{'...' if len(gaps) > 5 else ''}")
            else:
                log.info(f"{symbol}: bar data complete")

        # Check session labels for scored symbols only
        for symbol in self.symbols:
            self._check_and_backfill_labels(ch_client, symbol, trading_days)

    def _get_trading_days(
        self, ch_client, start_date: date, end_date: date
    ) -> list[date]:
        """Get list of trading days using SPY session_labels as calendar."""
        result = ch_client.query(
            """
            SELECT DISTINCT session_date
            FROM session_labels
            WHERE symbol         = 'SPY'
              AND window_minutes = 60
              AND session_date  >= %(start_date)s
              AND session_date  <= %(end_date)s
            ORDER BY session_date
            """,
            parameters={
                "start_date": start_date.isoformat(),
                "end_date":   end_date.isoformat(),
            }
        )
        days = [row[0] for row in result.result_rows]

        # Add today if it's a weekday and not already in the list
        if (end_date.weekday() < 5 and
            end_date not in days):
            days.append(end_date)

        return sorted(days)

    def _find_gaps(
        self, ch_client, symbol: str, trading_days: list[date]
    ) -> list[date]:
        """
        Find trading days where bar data is missing or incomplete.
        Returns list of dates that need backfill.
        """
        today      = date.today()
        start_date = min(trading_days)
        end_date   = today  # include today's partial session

        # Get actual bar counts per session
        result = ch_client.query(
            """
            SELECT
                toDate(toTimezone(ts, 'America/New_York'))  AS session_date,
                count()     AS bar_count
            FROM stock_bar FINAL
            WHERE symbol      = %(symbol)s
              AND period      = '1m'
              AND session     = 1
              AND toDate(toTimezone(ts, 'America/New_York')) >= %(start_date)s
              AND toDate(ts) <= %(end_date)s
            GROUP BY session_date
            ORDER BY session_date
            """,
            parameters={
                "symbol":     symbol,
                "start_date": start_date.isoformat(),
                "end_date":   end_date.isoformat(),
            }
        )

        actual = {row[0]: row[1] for row in result.result_rows}

        gaps = []
        now  = datetime.now(NY)

        for day in trading_days:
            bar_count = actual.get(day, 0)

            # For today's partial session, compute expected bars so far
            if day == today and now.hour < 16:
                minutes_elapsed = max(0,
                    (now.hour - 9) * 60 + now.minute - 30
                )
                expected = min(minutes_elapsed, 390)
                # Allow 10% tolerance for today's partial session
                min_bars = max(1, int(expected * 0.90))
            else:
                # Completed sessions: expect ~390 bars
                # Allow for early closes (minimum 200 bars)
                min_bars = MIN_BARS_PER_SESSION

            if bar_count < min_bars:
                gaps.append(day)

        return gaps

    # NOTE: _backfill_symbol() removed - MDS now handles bar backfill via BarGapDetector

    def _check_and_backfill_labels(
        self, ch_client, symbol: str, trading_days: list[date]
    ) -> None:
        """
        Check session_labels for completeness and compute any missing labels.
        Only covers completed sessions (not today).
        """
        from ml.etl.label_session import compute_and_write_label

        yesterday = date.today() - timedelta(days=1)
        while yesterday.weekday() >= 5:
            yesterday -= timedelta(days=1)

        completed_days = [d for d in trading_days if d <= yesterday]
        if not completed_days:
            return

        start_date = min(completed_days)

        result = ch_client.query(
            """
            SELECT DISTINCT session_date
            FROM session_labels
            WHERE symbol         = %(symbol)s
              AND window_minutes = 60
              AND session_date  >= %(start_date)s
            ORDER BY session_date
            """,
            parameters={
                "symbol":     symbol,
                "start_date": start_date.isoformat(),
            }
        )
        existing_labels = {row[0] for row in result.result_rows}

        missing = [d for d in completed_days if d not in existing_labels]
        if not missing:
            log.info(f"{symbol}: session labels complete")
            return

        log.info(f"{symbol}: computing {len(missing)} missing session labels")
        for session_date in missing:
            try:
                compute_and_write_label(ch_client, symbol, session_date)
            except Exception as e:
                log.warning(f"Failed to compute label for {symbol} "
                            f"{session_date}: {e}")

    def _score_from_history(self, session_date: date) -> None:
        """
        Score today's session using backfilled bars from ClickHouse.

        Called when scorer starts after the prediction window has passed.
        Fetches bars from ClickHouse, populates the BarStore, then scores
        each symbol using its configured window and publishes predictions.
        """
        ch = self._get_ch_client()

        # Fetch today's regular session bars for all required symbols
        all_symbols = list(set(self.symbols + CORRELATION_SYMBOLS))

        log.info(f"Loading bars from ClickHouse for {len(all_symbols)} symbols...")

        for symbol in all_symbols:
            result = ch.query(
                """
                SELECT ts, open, high, low, close, volume, vwap, trade_count, session
                FROM stock_bar FINAL
                WHERE symbol    = %(symbol)s
                  AND period    = '1m'
                  AND session   = 1
                  AND toDate(toTimezone(ts, 'America/New_York')) = %(session_date)s
                ORDER BY ts
                """,
                parameters={
                    "symbol":       symbol,
                    "session_date": session_date.isoformat(),
                }
            )

            bar_count = 0
            for row in result.result_rows:
                ts_raw, open_, high, low, close, volume, vwap, trade_count, session = row
                # Convert timestamp to datetime with NY timezone
                if isinstance(ts_raw, datetime):
                    ts = ts_raw.replace(tzinfo=NY) if ts_raw.tzinfo is None \
                         else ts_raw.astimezone(NY)
                else:
                    ts = datetime.fromisoformat(str(ts_raw)).astimezone(NY)

                bar = Bar(
                    ts          = ts,
                    open_       = float(open_),
                    high        = float(high),
                    low         = float(low),
                    close       = float(close),
                    volume      = int(volume),
                    vwap        = float(vwap) if vwap else 0.0,
                    trade_count = int(trade_count) if trade_count else 0,
                    session     = "regular",
                )
                self.bar_store.add_bar(symbol, bar)
                bar_count += 1

            if bar_count > 0:
                log.info(f"  {symbol}: loaded {bar_count} bars from ClickHouse")
            else:
                log.warning(f"  {symbol}: no bars found for {session_date}")

        # Now score each symbol using its configured window
        log.info("Scoring symbols from backfilled bars...")

        for symbol in self.symbols:
            if symbol in self._scored_today:
                continue

            cfg = self.pred_config.get(symbol)

            # Try primary window first
            window = cfg.window_minutes
            bar_count = self.bar_store.symbol_count(symbol)
            window_bars = self.bar_store.get_window_bars(symbol, session_date, window)

            if len(window_bars) < window - 5:
                # Not enough bars for this window, try fallback
                if cfg.fallback_window:
                    log.info(f"{symbol}: only {len(window_bars)} bars for w{window}, "
                             f"trying fallback w{cfg.fallback_window}")
                    window = cfg.fallback_window
                    window_bars = self.bar_store.get_window_bars(
                        symbol, session_date, window
                    )

            log.info(f"Scoring {symbol} at w{window} ({len(window_bars)} bars)")

            try:
                # Use lower threshold for historical scoring (0.5)
                prediction = self._score_symbol(symbol, session_date, window, 0.5)
                if prediction:
                    self._publish_prediction(symbol, prediction)
                    self._scored_today.add(symbol)
                else:
                    log.warning(f"{symbol}: scoring returned no prediction")
                    self._scored_today.add(symbol)  # Mark as done anyway
            except Exception as e:
                log.error(f"Historical scoring failed for {symbol}: {e}",
                          exc_info=True)
                self._scored_today.add(symbol)  # Mark as done to avoid retry

        scored_count = len(self._scored_today)
        log.info(f"Historical scoring complete: {scored_count}/{len(self.symbols)} symbols")

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
            self._pub.send_string(topic, zmq.SNDMORE)
            self._pub.send_string(payload)
            log.info(f"Published prediction for {symbol}: "
                     f"{prediction['prediction']} "
                     f"(confidence={prediction['confidence']:.3f})")
        # Log for outcome tracking
        self._log_prediction(symbol, prediction)

    def _score_symbol(
        self,
        symbol: str,
        today: date,
        window_minutes: int,
        min_confidence: float,
    ) -> Optional[dict]:
        """
        Compute features and score for one symbol at a specific window.

        Returns prediction dict if confidence >= min_confidence, else None.
        """
        model = self.models.get((symbol, window_minutes))
        if model is None:
            log.warning(f"No model loaded for {symbol} w{window_minutes}")
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
        w_bars = self.bar_store.get_window_bars(symbol, today, window_minutes)
        f15_bars = self.bar_store.get_window_bars(symbol, today, 15)

        # Compute feature vector
        features = compute_live_features(
            symbol         = symbol,
            today          = today,
            window_minutes = window_minutes,
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

        # Check if confidence meets threshold
        if confidence < min_confidence:
            log.info(f"{symbol} w{window_minutes}: confidence {confidence:.3f} < {min_confidence:.2f} threshold")
            return None

        # Compute window aggregates for context
        w_agg = bars_to_agg(w_bars)
        w_high = w_agg["high"]
        w_low = w_agg["low"]
        w_range = w_high - w_low
        open_930 = w_bars[0].open

        # Compute entry quality (how much opportunity remains)
        close_position = float(features["close_position"])
        entry_quality = close_position if label == 1 else (1.0 - close_position)

        # Build trade levels based on prediction
        if label == 1:  # fade_the_high
            entry_zone = (w_high - w_range * 0.1, w_high)
            target = w_low + w_range * 0.2
            stop = w_high + w_range * 0.15
        else:  # buy_the_dip
            entry_zone = (w_low, w_low + w_range * 0.1)
            target = w_high - w_range * 0.2
            stop = w_low - w_range * 0.15

        # Build message
        window_close = datetime(today.year, today.month, today.day,
                                9, 30, tzinfo=NY) + timedelta(minutes=window_minutes)
        market_close = datetime(today.year, today.month, today.day,
                                16, 0, tzinfo=NY)

        prediction = {
            "type":          "session_prediction",
            "symbol":        symbol,
            "timestamp":     window_close.isoformat(),
            "window_minutes": window_minutes,
            "prediction":    DIRECTIONAL_NAMES[label],
            "label":         label,
            "confidence":    round(confidence, 4),
            "entry_quality": round(entry_quality, 4),
            "model_version": f"{symbol.lower()}_directional_model_w{window_minutes}",
            "session_context": {
                "open_930":       round(open_930, 2),
                "first_hour_high": round(w_high, 2),
                "first_hour_low":  round(w_low, 2),
                "first_hour_range": round(w_range, 2),
                "close_position":  round(close_position, 4),
            },
            "trade_levels": {
                "entry_low":  round(entry_zone[0], 2),
                "entry_high": round(entry_zone[1], 2),
                "target":     round(target, 2),
                "stop":       round(stop, 2),
            },
            "validity": {
                "expires": market_close.isoformat(),
                "session_date": today.isoformat(),
            },
            "features":      {k: round(float(features[k]), 6)
                               for k in TOP_FEATURES_TO_PUBLISH
                               if k in features},
        }

        return prediction

    def _is_prediction_time(self, now: datetime, window_minutes: int) -> bool:
        """Check if it's time to score (within 30s after prediction window close)."""
        target_hour   = 9 + window_minutes // 60
        target_minute = 30 + window_minutes % 60
        if target_minute >= 60:
            target_hour += 1
            target_minute -= 60
        return (
            now.hour == target_hour and
            now.minute == target_minute and
            now.second < 30
        )

    def _get_unique_windows(self) -> set[int]:
        """Get all unique prediction windows needed for configured symbols."""
        windows = set()
        for sym in self.symbols:
            cfg = self.pred_config.get(sym)
            windows.add(cfg.window_minutes)
            if cfg.fallback_window:
                windows.add(cfg.fallback_window)
        return windows

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

        # Discover market data service first (needed for symbol list and bar subscription)
        log.info("Discovering market data service...")
        md_endpoint = ServiceLocator.wait_for_service(
            ServiceLocator.MARKET_DATA, timeout_sec=60
        )
        market_data_url = md_endpoint.pubSub if hasattr(md_endpoint, 'pubSub') \
                          else f"tcp://{md_endpoint.host}:6006"
        router_url = md_endpoint.router if hasattr(md_endpoint, 'router') \
                     else f"tcp://{md_endpoint.host}:6007"

        # Fetch symbol list from MDS
        log.info("Fetching trading universe from MDS...")
        try:
            self.symbols = fetch_symbol_list(router_url, "trading_universe")
            log.info(f"Trading universe: {self.symbols}")
        except RuntimeError as e:
            log.error(f"Failed to fetch symbol list: {e}")
            log.error("Cannot start scorer without symbol list")
            return

        # Run startup health check (backfill any missing bars/labels)
        log.info("Running startup health check...")
        ch = self._get_ch_client()
        self._startup_health_check(ch)
        log.info("Health check complete.")

        # Load models for all required windows per symbol
        for sym in self.symbols:
            cfg = self.pred_config.get(sym)
            windows_needed = [cfg.window_minutes]
            if cfg.fallback_window and cfg.fallback_window != cfg.window_minutes:
                windows_needed.append(cfg.fallback_window)

            for w in windows_needed:
                model = load_model(sym, w)
                if model:
                    self.models[(sym, w)] = model
                    log.info(f"Loaded model for {sym} w{w}")
                else:
                    log.warning(f"No model for {sym} w{w}")

        if not self.models and not self.dry_run:
            log.error("No models loaded -- exiting")
            return

        self._setup_zmq(market_data_url)

        # Start service advertiser
        self._advertiser = ServiceAdvertiser(
            service_name = SERVICE_NAME,
            pub_port     = PUB_PORT,
            router_port  = ROUTER_PORT,
        )
        self._advertiser.start()

        # Log per-symbol config
        for sym in self.symbols:
            cfg = self.pred_config.get(sym)
            fb = f" -> w{cfg.fallback_window}" if cfg.fallback_window else ""
            log.info(f"  {sym}: w{cfg.window_minutes} @ {cfg.confidence_threshold:.0%}{fb}")

        log.info(f"Session scorer started. Symbols: {self.symbols}  "
                 f"{'DRY RUN' if self.dry_run else 'LIVE'}")

        # Start bar ingestion in background thread
        threading.Thread(target=self._bar_ingestion_loop, daemon=True).start()

        # Run scoring loop in main thread
        self._scoring_loop()

    def _bar_ingestion_loop(self):
        """Continuously receive and store bars from ZMQ. Runs in background thread."""
        log.info("Bar ingestion thread started")
        while self._running:
            try:
                frames = self._sub.recv_multipart()
                if len(frames) >= 2:
                    topic = frames[0].decode("utf-8")
                    payload = frames[1].decode("utf-8")
                    self._handle_message(topic, payload)
            except zmq.Again:
                pass  # Timeout, loop continues
            except Exception as e:
                if self._running:
                    log.warning(f"Bar ingestion error: {e}")
        log.info("Bar ingestion thread stopped")

    def _scoring_loop(self):
        """Check scoring conditions and end-of-day. Can sleep freely."""
        today = date.today()

        while self._running:
            now = datetime.now(NY)

            # Reset state at start of new day
            if now.date() != today:
                today = now.date()
                self._scored_today.clear()
                self._early_attempted.clear()
                self.bar_store.clear()
                self._skip_today = False
                self._market_open = False
                self._market_open_time = None
                self._idle_logged = False
                self._eod_done = False
                log.info(f"New session: {today}")

            # Only process during market hours
            if not self._is_market_hours(now):
                time.sleep(5)
                continue

            # Late-start check: if we missed market open, score from backfilled bars
            market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
            if (self._market_open_time is None and
                now >= market_open_time and
                len(self._scored_today) == 0):
                # Assume market opened at 9:30 if we missed the open event
                self._market_open_time = market_open_time
                # Check if latest prediction window (w60 = 10:30) has passed
                latest_window = max(self._get_unique_windows())
                prediction_cutoff = self._market_open_time + timedelta(minutes=latest_window)
                if now > prediction_cutoff:
                    # Score from backfilled ClickHouse bars
                    log.info(f"Prediction window (w{latest_window}) passed -- "
                             "scoring from backfilled bars")
                    self._score_from_history(today)
                else:
                    log.info(f"Late start but prediction window still open "
                             f"(w{latest_window} at {prediction_cutoff.strftime('%H:%M')})")

            # Skip scoring if we missed market open
            if self._skip_today:
                time.sleep(1)
                continue

            # Check each unique window time
            for window in self._get_unique_windows():
                if not self._is_prediction_time(now, window):
                    continue

                # Find symbols whose primary window is this window (early-fire)
                for symbol in self.symbols:
                    if symbol in self._scored_today:
                        continue
                    cfg = self.pred_config.get(symbol)
                    if cfg.window_minutes != window:
                        continue
                    if symbol in self._early_attempted:
                        continue  # Already tried early, waiting for fallback

                    log.info(f"Scoring {symbol} at w{window} (early-fire, threshold={cfg.confidence_threshold})")
                    try:
                        prediction = self._score_symbol(
                            symbol, today, window, cfg.confidence_threshold
                        )
                        if prediction:
                            self._publish_prediction(symbol, prediction)
                            self._scored_today.add(symbol)
                        elif cfg.fallback_window:
                            log.info(f"{symbol}: early-fire below threshold, will retry at w{cfg.fallback_window}")
                            self._early_attempted.add(symbol)
                        else:
                            # No fallback, must publish anyway with lower threshold
                            log.info(f"{symbol}: no fallback, scoring with 0.5 threshold")
                            prediction = self._score_symbol(symbol, today, window, 0.5)
                            if prediction:
                                self._publish_prediction(symbol, prediction)
                            self._scored_today.add(symbol)
                    except Exception as e:
                        log.error(f"Scoring failed for {symbol}: {e}", exc_info=True)

                # Find symbols whose fallback window is this window
                for symbol in self.symbols:
                    if symbol in self._scored_today:
                        continue
                    if symbol not in self._early_attempted:
                        continue  # Didn't attempt early or doesn't have fallback
                    cfg = self.pred_config.get(symbol)
                    if cfg.fallback_window != window:
                        continue

                    log.info(f"Scoring {symbol} at w{window} (fallback)")
                    try:
                        # Fallback uses lower threshold (0.5) - always publish
                        prediction = self._score_symbol(symbol, today, window, 0.5)
                        if prediction:
                            self._publish_prediction(symbol, prediction)
                        self._scored_today.add(symbol)
                    except Exception as e:
                        log.error(f"Fallback scoring failed for {symbol}: {e}", exc_info=True)

            # Time-based EOD fallback: trigger at 4:05pm if cal.market.close was missed
            if (now.hour == 16 and now.minute >= 5 and not self._eod_done):
                log.info("4:05pm ET reached -- triggering EOD flush (calendar event fallback)")
                self._end_of_day()

            # If all symbols scored, idle until end of day
            if len(self._scored_today) == len(self.symbols):
                if not self._idle_logged:
                    log.info("All symbols scored. Idling until market close.")
                    self._idle_logged = True
                time.sleep(30)
            else:
                time.sleep(1)  # Check scoring conditions every second

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

            # Only store regular session bars
            if bar.session != "regular":
                return

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
                self._end_of_day()

        except (json.JSONDecodeError, KeyError) as e:
            log.debug(f"Calendar event parse error: {e}")

    def _end_of_day(self):
        """
        End-of-day processing.

        Called when market close event is received.
        Bar writes are handled by MDS (EquityBarService) — the scorer
        does not write bars to ClickHouse.
        """
        if self._eod_done:
            return
        self._eod_done = True

        today = date.today()

        # Update prediction outcomes
        self._update_prediction_outcomes(today)

    def _log_prediction(self, symbol: str, prediction: dict):
        """
        Log a prediction for later outcome tracking.

        Stores prediction in memory; outcome is updated at EOD.
        """
        self._prediction_log[symbol] = {
            "prediction": prediction.copy(),
            "logged_at": datetime.now(NY).isoformat(),
        }
        log.debug(f"Logged prediction for {symbol}")

    def _update_prediction_outcomes(self, session_date: date):
        """
        Update prediction outcomes at end of day.

        Computes actual session high/low, determines if prediction was correct,
        and writes results to ClickHouse prediction_log table.
        """
        if not self._prediction_log:
            return

        ch = self._get_ch_client()
        rows = []

        for symbol, entry in self._prediction_log.items():
            pred = entry["prediction"]
            bars = self.bar_store.get_bars(symbol)

            if not bars:
                log.warning(f"No bars for {symbol} -- cannot compute outcome")
                continue

            # Compute session high/low from all bars
            session_high = max(b.high for b in bars)
            session_low = min(b.low for b in bars)
            session_close = bars[-1].close

            # Get first-hour levels from prediction
            ctx = pred.get("session_context", {})
            fh_high = ctx.get("first_hour_high", 0)
            fh_low = ctx.get("first_hour_low", 0)

            # Determine actual outcome
            fh_high_is_session_high = abs(fh_high - session_high) < 0.01
            fh_low_is_session_low = abs(fh_low - session_low) < 0.01

            # Predicted direction
            predicted_label = pred.get("label", -1)

            # Actual direction (1 = fade_the_high, 0 = buy_the_dip)
            if fh_high_is_session_high and not fh_low_is_session_low:
                actual_label = 1  # fade_the_high
            elif fh_low_is_session_low and not fh_high_is_session_high:
                actual_label = 0  # buy_the_dip
            else:
                actual_label = -1  # not a clean reversal

            correct = (predicted_label == actual_label) if actual_label >= 0 else None

            rows.append([
                symbol,
                session_date,
                pred.get("window_minutes", 60),
                predicted_label,
                pred.get("prediction", ""),
                pred.get("confidence", 0),
                actual_label,
                DIRECTIONAL_NAMES.get(actual_label, "none"),
                1 if correct else (0 if correct is False else None),
                fh_high,
                fh_low,
                session_high,
                session_low,
                session_close,
                pred.get("timestamp", ""),
            ])

            outcome_str = "CORRECT" if correct else ("WRONG" if correct is False else "N/A")
            log.info(f"  {symbol}: predicted={pred.get('prediction')} "
                     f"actual={DIRECTIONAL_NAMES.get(actual_label, 'none')} "
                     f"outcome={outcome_str}")

        if rows:
            try:
                ch.insert(
                    "prediction_log",
                    rows,
                    column_names=[
                        "symbol", "session_date", "window_minutes",
                        "predicted_label", "predicted_direction", "confidence",
                        "actual_label", "actual_direction", "correct",
                        "fh_high", "fh_low", "session_high", "session_low",
                        "session_close", "prediction_timestamp",
                    ],
                )
                log.info(f"Logged {len(rows)} prediction outcomes to ClickHouse")
            except Exception as e:
                log.warning(f"Failed to write prediction outcomes: {e}")

        self._prediction_log.clear()

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
    parser.add_argument("--dry-run",   action="store_true",
                        help="Compute and log predictions without publishing")
    args = parser.parse_args()

    scorer = SessionScorer(
        dry_run = args.dry_run,
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
