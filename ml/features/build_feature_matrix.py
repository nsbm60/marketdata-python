"""
build_feature_matrix.py

Compute session-level features and labels from 1-minute OHLCV bars in ClickHouse.
Outputs one row per trading session as a CSV file.

Usage:
    python build_feature_matrix.py --symbol NVDA
    python build_feature_matrix.py --symbol NVDA --start 2023-01-01 --end 2024-12-31
    python build_feature_matrix.py --symbol NVDA --out /path/to/output.csv

Dependencies:
    pip install clickhouse-connect pandas python-dotenv pyzmq

Environment variables:
    CLICKHOUSE_USER       (default: default)
    CLICKHOUSE_PASSWORD   (default: Aector99)
    CLICKHOUSE_DATABASE   (default: trading)

ClickHouse host/port are discovered automatically via ZMQ discovery bus.
"""

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

# Add project root to path for discovery module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import clickhouse_connect
import numpy as np
import pandas as pd
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
# SQL: one row per session with raw aggregates
# All time filtering is done here; feature math happens in Python below.
# ---------------------------------------------------------------------------

SESSION_AGG_SQL = """
WITH

-- Regular session bars only
regular AS (
    SELECT
        toDate(ts)              AS session_date,
        ts,
        open, high, low, close,
        volume,
        vwap,
        toHour(ts)              AS hr,
        toMinute(ts)            AS mn
    FROM stock_bars_1m
    WHERE symbol    = %(symbol)s
      AND session   = 1              -- regular session only
      AND toDate(ts) BETWEEN %(start)s AND %(end)s
),

-- Full session aggregates
full_session AS (
    SELECT
        session_date,
        argMin(open,  ts)       AS open_930,
        argMax(close, ts)       AS close_400,
        max(high)               AS session_high,
        min(low)                AS session_low,
        sum(volume)             AS session_volume
    FROM regular
    GROUP BY session_date
),

-- First hour: 9:30 through 10:29 inclusive (hr=9 OR hr=10,mn=0..29 --
-- actually 9:30-10:30 means bars where ts < 10:30, i.e. hr=9 or (hr=10 and mn < 30))
-- We use 9:30-10:30 window (60 minutes)
first_hour AS (
    SELECT
        session_date,
        max(high)               AS fh_high,
        min(low)                AS fh_low,
        argMax(close, ts)       AS fh_close,     -- close of 10:29 bar
        sum(volume)             AS fh_volume,
        sum(volume * vwap) / sum(volume) AS fh_vwap
    FROM regular
    WHERE (hr = 9) OR (hr = 10 AND mn < 30)
    GROUP BY session_date
),

-- First 15 minutes: 9:30 through 9:44 inclusive
first_15 AS (
    SELECT
        session_date,
        max(high)               AS f15_high,
        min(low)                AS f15_low,
        argMax(close, ts)       AS f15_close,
        sum(volume)             AS f15_volume
    FROM regular
    WHERE hr = 9 AND mn < 45    -- 9:30..9:44
    GROUP BY session_date
),

-- Prior session close (last bar of each session)
-- We lag this: for each session_date, we want the close of the PREVIOUS session
prior_close AS (
    SELECT
        session_date,
        close_400,
        session_high,
        session_low,
        session_volume,
        lagInFrame(close_400)   OVER (ORDER BY session_date) AS prev_close,
        lagInFrame(session_high) OVER (ORDER BY session_date) AS prev_high,
        lagInFrame(session_low)  OVER (ORDER BY session_date) AS prev_low,
        lagInFrame(session_volume) OVER (ORDER BY session_date) AS prev_volume
    FROM full_session
)

SELECT
    fs.session_date,
    fs.open_930,
    fs.close_400,
    fs.session_high,
    fs.session_low,
    fs.session_volume,
    pc.prev_close,
    pc.prev_high,
    pc.prev_low,
    pc.prev_volume,
    fh.fh_high,
    fh.fh_low,
    fh.fh_close,
    fh.fh_volume,
    fh.fh_vwap,
    f15.f15_high,
    f15.f15_low,
    f15.f15_close,
    f15.f15_volume
FROM full_session     fs
JOIN first_hour       fh  ON fs.session_date = fh.session_date
JOIN first_15         f15 ON fs.session_date = f15.session_date
JOIN prior_close      pc  ON fs.session_date = pc.session_date
WHERE pc.prev_close IS NOT NULL   -- drop first session (no prior close)
ORDER BY fs.session_date
"""


# ---------------------------------------------------------------------------
# Feature computation (Python / pandas)
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw session aggregates from SQL and derives ML features.
    All features use only information available by 10:30am on the session date.
    Labels use full-session data (retrospective).
    """

    out = pd.DataFrame()
    out["date"] = df["session_date"]

    # --- Gap features (known at 9:30) ---

    # Gap as % of prior close.  Positive = gap up, negative = gap down.
    out["gap_pct"] = (df["open_930"] - df["prev_close"]) / df["prev_close"]

    # Prior day range normalized by prior close -- how volatile was yesterday?
    out["prior_day_range_pct"] = (df["prev_high"] - df["prev_low"]) / df["prev_close"]

    # --- ATR20: 20-session rolling average of (high - low) ---
    # Computed from session_high/low of prior sessions (no lookahead)
    daily_range = df["session_high"] - df["session_low"]
    # Shift by 1 so today's range doesn't contaminate today's ATR
    out["atr20"] = daily_range.shift(1).rolling(20, min_periods=5).mean()

    # --- First-hour range features (known at 10:30) ---

    fh_range = df["fh_high"] - df["fh_low"]
    out["fh_range_abs"]  = fh_range
    out["fh_range_pct"]  = fh_range / df["open_930"]          # normalized by open
    out["fh_range_atr"]  = fh_range / out["atr20"]            # normalized by ATR20

    # VWAP deviation: where did the first hour close relative to its own VWAP?
    # Positive = closed above VWAP (bullish), negative = below (bearish)
    out["fh_vwap_dev"] = (df["fh_close"] - df["fh_vwap"]) / df["fh_vwap"]

    # --- First-15-minute features (known at 9:45) ---

    f15_range = df["f15_high"] - df["f15_low"]
    # First 15 range as fraction of first hour range --
    # high ratio = most of the day's volatility front-loaded, typical of sweep
    out["f15_range_ratio"] = f15_range / fh_range.replace(0, float("nan"))

    # First 15 volume as fraction of first hour volume --
    # high ratio = volume front-loaded, another sweep indicator
    out["f15_vol_ratio"] = df["f15_volume"] / df["fh_volume"].replace(0, float("nan"))

    # --- Volume features ---

    # 20-day rolling average daily volume (prior sessions, no lookahead)
    avg_vol_20 = df["session_volume"].shift(1).rolling(20, min_periods=5).mean()
    out["avg_vol_20"] = avg_vol_20

    # First hour volume as multiple of average daily volume.
    # > 0.4 = elevated (first hour normally ~25-30% of daily)
    out["fh_vol_ratio"] = df["fh_volume"] / avg_vol_20

    # --- Sweep signal ---
    # Definition: first 15 minutes makes a significant directional move from open,
    # but the first hour CLOSES on the opposite side of open.
    # This is the stop-hunt signature.
    #
    # "Significant" = f15 extreme is more than 25% of fh_range away from open
    open_930 = df["open_930"]
    f15_up_extent   = (df["f15_high"] - open_930)              # how far above open
    f15_down_extent = (open_930 - df["f15_low"])               # how far below open

    significance_threshold = fh_range * 0.25

    # Upside sweep: price ran up significantly in first 15, then closed below open
    sweep_up   = (f15_up_extent   > significance_threshold) & (df["fh_close"] < open_930)
    # Downside sweep: price ran down significantly in first 15, then closed above open
    sweep_down = (f15_down_extent > significance_threshold) & (df["fh_close"] > open_930)

    out["sweep_signal"] = (sweep_up | sweep_down).astype(int)
    out["sweep_direction"] = 0  # 0=none, 1=up sweep (fakeout high), -1=down sweep (fakeout low)
    out.loc[sweep_up,   "sweep_direction"] =  1
    out.loc[sweep_down, "sweep_direction"] = -1

    # --- Label computation (uses full session data -- retrospective only) ---
    #
    # Tolerance: first-hour extreme counts as "session extreme" if within 0.1% of
    # the true session extreme. Handles floating point and micro-violations.
    tol = 0.001

    fh_high_is_session_high = df["fh_high"] >= df["session_high"] * (1 - tol)
    fh_low_is_session_low   = df["fh_low"]  <= df["session_low"]  * (1 + tol)

    out["fh_high_is_session_high"] = fh_high_is_session_high.astype(int)
    out["fh_low_is_session_low"]   = fh_low_is_session_low.astype(int)

    # Session extension beyond first-hour range (as multiple of fh_range)
    # Used to distinguish trend days from containment days
    upside_extension   = (df["session_high"] - df["fh_high"]) / fh_range.replace(0, float("nan"))
    downside_extension = (df["fh_low"] - df["session_low"])   / fh_range.replace(0, float("nan"))
    max_extension = upside_extension.combine(downside_extension, max)

    # Label:
    #   3 = Double sweep: first hour captures both session extremes (chop / pin)
    #   2 = Reversal: one first-hour extreme is session extreme (stop hunt then trend)
    #   1 = Containment: session stays close to first-hour range (no extension)
    #   0 = Trend: session extends significantly beyond first-hour range

    TREND_EXTENSION_THRESHOLD = 0.75  # session extends > 75% of fh_range beyond it

    conditions = [
        fh_high_is_session_high & fh_low_is_session_low,          # 3: double sweep
        fh_high_is_session_high | fh_low_is_session_low,          # 2: reversal (one side)
        max_extension <= TREND_EXTENSION_THRESHOLD,                # 1: containment
    ]
    choices = [3, 2, 1]
    out["label"] = pd.Series(
        np.select(conditions, choices, default=0),
        index=df.index
    )

    # Readable label name for inspection
    label_names = {0: "trend", 1: "containment", 2: "reversal", 3: "double_sweep"}
    out["label_name"] = out["label"].map(label_names)

    # Drop rows where ATR or avg_vol couldn't be computed (insufficient history)
    out = out.dropna(subset=["atr20", "avg_vol_20"])

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build ML feature matrix from ClickHouse bars")
    parser.add_argument("--symbol",  required=True,  help="Ticker symbol, e.g. NVDA")
    parser.add_argument("--start",   default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=date.today().isoformat(), help="End date YYYY-MM-DD")
    parser.add_argument("--out",     default=None, help="Output CSV path (default: <symbol>_features.csv)")
    args = parser.parse_args()

    symbol   = args.symbol.upper()
    out_path = args.out or f"{symbol.lower()}_features.csv"

    # --- ClickHouse client (discovered via ZMQ) ---
    log.info("Discovering ClickHouse...")
    ch_endpoint = ServiceLocator.wait_for_service(
        ServiceLocator.CLICKHOUSE,
        timeout_sec=60,
    )

    ch_client = clickhouse_connect.get_client(
        host     = ch_endpoint.host,
        port     = ch_endpoint.port,
        username = os.environ.get("CLICKHOUSE_USER",     "default"),
        password = os.environ.get("CLICKHOUSE_PASSWORD", "Aector99"),
        database = os.environ.get("CLICKHOUSE_DATABASE", "trading"),
    )

    log.info(f"Querying session aggregates for {symbol} ({args.start} → {args.end})")
    result = ch_client.query(
        SESSION_AGG_SQL,
        parameters={"symbol": symbol, "start": args.start, "end": args.end},
    )
    ch_client.close()

    df_raw = pd.DataFrame(result.result_rows, columns=result.column_names)
    log.info(f"  {len(df_raw)} sessions retrieved from ClickHouse")

    if df_raw.empty:
        log.error("No data returned — check symbol and date range")
        sys.exit(1)

    log.info("Computing features and labels...")
    df_features = compute_features(df_raw)

    log.info(f"  {len(df_features)} sessions after dropping warm-up rows")

    # Summary
    label_counts = df_features["label_name"].value_counts()
    log.info("Label distribution:")
    for name, count in label_counts.items():
        pct = 100 * count / len(df_features)
        log.info(f"  {name:<15} {count:>4}  ({pct:.1f}%)")

    df_features.to_csv(out_path, index=False)
    log.info(f"Written to {out_path}")


if __name__ == "__main__":
    main()
