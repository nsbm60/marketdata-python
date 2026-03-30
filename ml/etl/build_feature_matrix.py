"""
build_feature_matrix.py

Compute session-level features and labels from 1-minute OHLCV bars in ClickHouse.
Supports variable prediction windows: 15, 30, 45, or 60 minutes from open.

The window determines the prediction cutoff time:
  --window 15  ->  features known by 9:45am  (first 15 minutes only)
  --window 30  ->  features known by 10:00am (first 30 minutes)
  --window 45  ->  features known by 10:15am (first 45 minutes)
  --window 60  ->  features known by 10:30am (full first hour, default)

Use --all-windows to produce all four CSVs in a single run, for comparison.

Usage:
    python build_feature_matrix.py --symbol NVDA
    python build_feature_matrix.py --symbol NVDA --window 30
    python build_feature_matrix.py --symbol NVDA --all-windows
    python build_feature_matrix.py --symbol NVDA --start 2021-01-01

Dependencies:
    pip install clickhouse-connect pandas numpy python-dotenv pyzmq

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

VALID_WINDOWS = [15, 30, 45, 60]

# Rolling window for correlation features (sessions, not days)
CORR_WINDOW = 60


SESSION_AGG_SQL = """
WITH

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
      AND session   = 1
      AND toDate(ts) BETWEEN %(start)s AND %(end)s
),

full_session AS (
    SELECT
        session_date,
        argMin(open,  ts)           AS open_930,
        argMax(close, ts)           AS close_400,
        max(high)                   AS session_high,
        min(low)                    AS session_low,
        sum(volume)                 AS session_volume
    FROM regular
    GROUP BY session_date
),

w15 AS (
    SELECT
        session_date,
        max(high)                           AS w15_high,
        min(low)                            AS w15_low,
        argMax(close, ts)                   AS w15_close,
        sum(volume)                         AS w15_volume,
        sum(volume * vwap) / sum(volume)    AS w15_vwap,
        dateDiff('minute',
            toDateTime(session_date) + toIntervalHour(9) + toIntervalMinute(30),
            argMax(ts, high))               AS mins_to_w15_high,
        dateDiff('minute',
            toDateTime(session_date) + toIntervalHour(9) + toIntervalMinute(30),
            argMin(ts, low))                AS mins_to_w15_low
    FROM regular
    WHERE hr = 9 AND mn < 45
    GROUP BY session_date
),

w30 AS (
    SELECT
        session_date,
        max(high)                           AS w30_high,
        min(low)                            AS w30_low,
        argMax(close, ts)                   AS w30_close,
        sum(volume)                         AS w30_volume,
        sum(volume * vwap) / sum(volume)    AS w30_vwap
    FROM regular
    WHERE hr = 9
    GROUP BY session_date
),

w45 AS (
    SELECT
        session_date,
        max(high)                           AS w45_high,
        min(low)                            AS w45_low,
        argMax(close, ts)                   AS w45_close,
        sum(volume)                         AS w45_volume,
        sum(volume * vwap) / sum(volume)    AS w45_vwap
    FROM regular
    WHERE hr = 9 OR (hr = 10 AND mn < 15)
    GROUP BY session_date
),

w60 AS (
    SELECT
        session_date,
        max(high)                           AS w60_high,
        min(low)                            AS w60_low,
        argMax(close, ts)                   AS w60_close,
        sum(volume)                         AS w60_volume,
        sum(volume * vwap) / sum(volume)    AS w60_vwap,
        dateDiff('minute',
            toDateTime(session_date) + toIntervalHour(9) + toIntervalMinute(30),
            argMax(ts, high))               AS mins_to_w60_high,
        dateDiff('minute',
            toDateTime(session_date) + toIntervalHour(9) + toIntervalMinute(30),
            argMin(ts, low))                AS mins_to_w60_low
    FROM regular
    WHERE hr = 9 OR (hr = 10 AND mn < 30)
    GROUP BY session_date
),

prior AS (
    SELECT
        session_date,
        lagInFrame(close_400)      OVER (ORDER BY session_date) AS prev_close,
        lagInFrame(session_high)   OVER (ORDER BY session_date) AS prev_high,
        lagInFrame(session_low)    OVER (ORDER BY session_date) AS prev_low,
        lagInFrame(session_volume) OVER (ORDER BY session_date) AS prev_volume
    FROM full_session
)

SELECT
    fs.session_date         AS session_date,
    fs.open_930             AS open_930,
    fs.close_400            AS close_400,
    fs.session_high         AS session_high,
    fs.session_low          AS session_low,
    fs.session_volume       AS session_volume,
    p.prev_close            AS prev_close,
    p.prev_high             AS prev_high,
    p.prev_low              AS prev_low,
    p.prev_volume           AS prev_volume,
    w15.w15_high            AS w15_high,
    w15.w15_low             AS w15_low,
    w15.w15_close           AS w15_close,
    w15.w15_volume          AS w15_volume,
    w15.w15_vwap            AS w15_vwap,
    w15.mins_to_w15_high    AS mins_to_w15_high,
    w15.mins_to_w15_low     AS mins_to_w15_low,
    w30.w30_high            AS w30_high,
    w30.w30_low             AS w30_low,
    w30.w30_close           AS w30_close,
    w30.w30_volume          AS w30_volume,
    w30.w30_vwap            AS w30_vwap,
    w45.w45_high            AS w45_high,
    w45.w45_low             AS w45_low,
    w45.w45_close           AS w45_close,
    w45.w45_volume          AS w45_volume,
    w45.w45_vwap            AS w45_vwap,
    w60.w60_high            AS w60_high,
    w60.w60_low             AS w60_low,
    w60.w60_close           AS w60_close,
    w60.w60_volume          AS w60_volume,
    w60.w60_vwap            AS w60_vwap,
    w60.mins_to_w60_high    AS mins_to_w60_high,
    w60.mins_to_w60_low     AS mins_to_w60_low
FROM full_session fs
JOIN w15  ON fs.session_date = w15.session_date
JOIN w30  ON fs.session_date = w30.session_date
JOIN w45  ON fs.session_date = w45.session_date
JOIN w60  ON fs.session_date = w60.session_date
JOIN prior p ON fs.session_date = p.session_date
WHERE p.prev_close IS NOT NULL
ORDER BY fs.session_date
"""


# Correlation SQL: daily closes for QQQ and SMH
# We pull extra history (lookback buffer) so rolling correlation is available
# from the start of the target symbol's date range.
CORRELATION_SQL = """
SELECT
    symbol,
    toDate(ts)          AS session_date,
    argMax(close, ts)   AS close
FROM stock_bars_1m
WHERE symbol  IN ('QQQ', 'SMH')
  AND session = 1
  AND toDate(ts) BETWEEN %(start)s AND %(end)s
GROUP BY symbol, session_date
ORDER BY symbol, session_date
"""


def fetch_correlation_features(ch_client, start: str, end: str) -> pd.DataFrame:
    """
    Compute rolling 60-session correlations of the target symbol with QQQ and SMH.
    Returns a DataFrame indexed by session_date with columns:
        target_qqq_corr_60  -- rolling 60-session correlation with QQQ (lagged 1)
        target_smh_corr_60  -- rolling 60-session correlation with SMH (lagged 1)
        target_qqq_beta_60  -- rolling 60-session beta to QQQ (lagged 1)

    Note: the correlation is computed from daily close returns, not intraday.
    All values are lagged by 1 session so there is no lookahead.

    We pull extra history before the start date to ensure the rolling window
    is populated from session 1 of the target range.
    """
    # Pull from well before start to warm up the rolling window
    import datetime
    start_dt     = datetime.date.fromisoformat(start)
    buffer_start = (start_dt - datetime.timedelta(days=120)).isoformat()

    result = ch_client.query(
        CORRELATION_SQL,
        parameters={"start": buffer_start, "end": end},
    )

    df = pd.DataFrame(result.result_rows, columns=["symbol", "session_date", "close"])
    df["session_date"] = pd.to_datetime(df["session_date"])
    df = df.pivot(index="session_date", columns="symbol", values="close").sort_index()

    missing = [s for s in ["QQQ", "SMH"] if s not in df.columns]
    if missing:
        log.warning(f"Correlation symbols missing from ClickHouse: {missing}. "
                    f"Skipping correlation features.")
        return pd.DataFrame()

    # Daily returns
    rets = df.pct_change().dropna()

    # We need a target symbol return series -- build from the session_date index
    # The caller will supply NVDA returns separately via merge.
    # Return the benchmark returns so compute_features can merge them in.
    return rets.rename(columns={"QQQ": "qqq_ret", "SMH": "smh_ret"})


def compute_features(df: pd.DataFrame,
                     window: int,
                     benchmark_rets: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute session features using data available up to the prediction window.

    window:          minutes from open (15, 30, 45, or 60)
    benchmark_rets:  DataFrame with columns [qqq_ret, smh_ret] indexed by date,
                     from fetch_correlation_features(). Optional -- if None,
                     correlation features are omitted.

    Labels are always defined relative to the full first hour (w60) extremes,
    keeping them consistent across window comparisons.
    """
    assert window in VALID_WINDOWS

    w        = f"w{window}"
    out      = pd.DataFrame()
    out["date"]   = pd.to_datetime(df["session_date"])
    out["window"] = window
    open_930 = df["open_930"]

    # --- Pre-open features (known at 9:30, same for all windows) ---

    out["gap_pct"]             = (open_930 - df["prev_close"]) / df["prev_close"]
    out["prior_day_range_pct"] = (df["prev_high"] - df["prev_low"]) / df["prev_close"]

    daily_range  = df["session_high"] - df["session_low"]
    atr20        = daily_range.shift(1).rolling(20, min_periods=5).mean()
    out["atr20"] = atr20

    # --- Window price/volume features ---

    w_high   = df[f"{w}_high"]
    w_low    = df[f"{w}_low"]
    w_close  = df[f"{w}_close"]
    w_volume = df[f"{w}_volume"]
    w_vwap   = df[f"{w}_vwap"]
    w_range  = w_high - w_low

    # Raw price levels (needed for stage 2 models)
    out["open_930"] = open_930
    out["w_high"]   = w_high
    out["w_low"]    = w_low

    out["w_range_pct"] = w_range / open_930
    out["w_range_atr"] = w_range / atr20
    out["w_vwap_dev"]  = (w_close - w_vwap) / w_vwap

    avg_vol_20         = df["session_volume"].shift(1).rolling(20, min_periods=5).mean()
    out["avg_vol_20"]  = avg_vol_20
    out["w_vol_ratio"] = w_volume / avg_vol_20

    # --- First-15-minute sub-features ---

    f15_high   = df["w15_high"]
    f15_low    = df["w15_low"]
    f15_volume = df["w15_volume"]
    f15_range  = f15_high - f15_low

    out["f15_range_ratio"] = f15_range / w_range.replace(0, float("nan"))
    out["f15_vol_ratio"]   = f15_volume / w_volume.replace(0, float("nan"))

    # --- Sweep signal ---

    f15_up_ext   = f15_high - open_930
    f15_down_ext = open_930 - f15_low
    significance = w_range * 0.25

    sweep_up   = (f15_up_ext   > significance) & (w_close < open_930)
    sweep_down = (f15_down_ext > significance) & (w_close > open_930)

    out["sweep_signal"]    = (sweep_up | sweep_down).astype(int)
    out["sweep_direction"] = 0
    out.loc[sweep_up,   "sweep_direction"] =  1
    out.loc[sweep_down, "sweep_direction"] = -1

    # --- Time-of-extreme features ---

    if window in (15, 60):
        out["mins_to_high"] = df[f"mins_to_{w}_high"]
        out["mins_to_low"]  = df[f"mins_to_{w}_low"]
        out["high_timing"]  = df[f"mins_to_{w}_high"] / window
        out["low_timing"]   = df[f"mins_to_{w}_low"]  / window
    else:
        out["mins_to_high"] = np.nan
        out["mins_to_low"]  = np.nan
        out["high_timing"]  = np.nan
        out["low_timing"]   = np.nan

    # --- Confirmation signal features ---

    out["reversal_progress"] = (w_high - w_close) / w_range.replace(0, float("nan"))
    out["close_position"]    = (w_close - w_low)  / w_range.replace(0, float("nan"))

    # --- Regime features (rolling, lagged -- no lookahead) ---

    tol      = 0.001
    w60_high = df["w60_high"]
    w60_low  = df["w60_low"]

    fh_high_is_session_high = w60_high >= df["session_high"] * (1 - tol)
    fh_low_is_session_low   = w60_low  <= df["session_low"]  * (1 + tol)

    is_reversal = (fh_high_is_session_high | fh_low_is_session_low).astype(int)

    out["rolling_reversal_rate"] = (
        is_reversal.shift(1).rolling(30, min_periods=15).mean()
    )
    out["rolling_high_set_rate"] = (
        fh_high_is_session_high.astype(int).shift(1).rolling(30, min_periods=15).mean()
    )
    out["directional_bias"]     = (out["rolling_high_set_rate"] - 0.5) * 2
    out["gap_regime_alignment"] = out["gap_pct"] * out["directional_bias"]

    # --- Correlation / market regime features ---
    # Rolling 60-session correlation with QQQ and SMH.
    # High correlation = macro-driven environment.
    # Low correlation  = idiosyncratic / stock-specific narrative.
    # All values lagged by 1 session -- no lookahead.

    if benchmark_rets is not None and not benchmark_rets.empty:
        # Compute target symbol daily returns from session closes
        target_ret = df["close_400"].pct_change()
        target_ret.index = pd.to_datetime(df["session_date"].values)

        # Merge benchmark returns onto target dates
        merged = pd.DataFrame({"target": target_ret.values},
                               index=pd.to_datetime(df["session_date"].values))
        merged = merged.join(benchmark_rets, how="left")

        # Rolling correlation -- shift(1) for no lookahead
        target_qqq_corr = (
            merged["target"].rolling(CORR_WINDOW, min_periods=20)
            .corr(merged["qqq_ret"])
            .shift(1)
        )
        target_smh_corr = (
            merged["target"].rolling(CORR_WINDOW, min_periods=20)
            .corr(merged["smh_ret"])
            .shift(1)
        )

        # Beta to QQQ: covariance / variance (60-session rolling)
        def rolling_beta(y, x, window):
            cov = y.rolling(window, min_periods=20).cov(x)
            var = x.rolling(window, min_periods=20).var()
            return (cov / var).shift(1)

        target_qqq_beta = rolling_beta(merged["target"], merged["qqq_ret"], CORR_WINDOW)

        # Correlation regime: how far is current correlation from its own
        # 252-session mean? Positive = more correlated than usual (macro fear).
        corr_mean_252 = target_qqq_corr.rolling(252, min_periods=60).mean()
        corr_dev      = target_qqq_corr - corr_mean_252

        out["target_qqq_corr"]     = target_qqq_corr.values
        out["target_smh_corr"]     = target_smh_corr.values
        out["target_qqq_beta"]     = target_qqq_beta.values
        out["corr_regime_dev"]   = corr_dev.values  # deviation from long-run mean

        log.info("  Correlation features added.")
    else:
        out["target_qqq_corr"]   = np.nan
        out["target_smh_corr"]   = np.nan
        out["target_qqq_beta"]   = np.nan
        out["corr_regime_dev"] = np.nan
        log.warning("  Correlation features unavailable (QQQ/SMH not in ClickHouse).")

    # --- Labels (full session, retrospective -- same for all windows) ---

    w60_range = w60_high - w60_low

    out["fh_high_is_session_high"] = fh_high_is_session_high.astype(int)
    out["fh_low_is_session_low"]   = fh_low_is_session_low.astype(int)

    upside_ext   = (df["session_high"] - w60_high) / w60_range.replace(0, float("nan"))
    downside_ext = (w60_low - df["session_low"])   / w60_range.replace(0, float("nan"))
    max_ext      = upside_ext.combine(downside_ext, max)

    conditions = [
        fh_high_is_session_high & fh_low_is_session_low,
        fh_high_is_session_high | fh_low_is_session_low,
        max_ext <= 0.75,
    ]
    out["label"]        = pd.Series(np.select(conditions, [3, 2, 1], default=0),
                                     index=df.index)
    out["label_name"]   = out["label"].map(
        {0: "trend", 1: "containment", 2: "reversal", 3: "double_sweep"}
    )
    out["binary_label"] = out["label"].isin([2, 3]).astype(int)

    directional = pd.Series(np.nan, index=df.index)
    directional[fh_high_is_session_high & ~fh_low_is_session_low] = 1
    directional[fh_low_is_session_low   & ~fh_high_is_session_high] = 0
    out["directional_label"] = directional

    out = out.dropna(subset=["atr20", "avg_vol_20"])

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build ML feature matrix from ClickHouse bars"
    )
    parser.add_argument("--symbol",      required=True,  help="Ticker symbol, e.g. NVDA")
    parser.add_argument("--start",       default="2021-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",         default=date.today().isoformat(), help="End date YYYY-MM-DD")
    parser.add_argument("--window",      type=int, default=60, choices=VALID_WINDOWS,
                        help="Prediction window in minutes (default: 60)")
    parser.add_argument("--all-windows", action="store_true",
                        help="Produce CSVs for all four windows in one run")
    parser.add_argument("--out",         default=None,
                        help="Output CSV path (ignored with --all-windows)")
    parser.add_argument("--out-dir",     default="data",
                        help="Output directory (default: data)")
    parser.add_argument("--no-corr",     action="store_true",
                        help="Skip correlation features (faster, for quick iteration)")
    args = parser.parse_args()

    symbol  = args.symbol.upper()
    windows = VALID_WINDOWS if args.all_windows else [args.window]

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

    # --- Fetch correlation benchmarks (QQQ, SMH) ---
    benchmark_rets = None
    if not args.no_corr:
        log.info("Fetching QQQ/SMH correlation data...")
        benchmark_rets = fetch_correlation_features(ch_client, args.start, args.end)
        if benchmark_rets.empty:
            log.warning("No correlation data available -- continuing without it.")

    # --- Fetch target symbol session aggregates ---
    log.info(f"Querying session aggregates for {symbol} ({args.start} -> {args.end})")
    result = ch_client.query(
        SESSION_AGG_SQL,
        parameters={"symbol": symbol, "start": args.start, "end": args.end},
    )
    ch_client.close()

    df_raw = pd.DataFrame(result.result_rows, columns=result.column_names)
    log.info(f"  {len(df_raw)} sessions retrieved from ClickHouse")

    if df_raw.empty:
        log.error("No data returned -- check symbol and date range")
        sys.exit(1)

    for w in windows:
        log.info(f"Computing features for window={w}min...")
        df_features = compute_features(df_raw, window=w, benchmark_rets=benchmark_rets)
        log.info(f"  {len(df_features)} sessions after dropping warm-up rows")

        label_counts = df_features["label_name"].value_counts()
        log.info("  Label distribution:")
        for name, count in label_counts.items():
            pct = 100 * count / len(df_features)
            log.info(f"    {name:<15} {count:>4}  ({pct:.1f}%)")

        # Correlation feature summary
        if "target_qqq_corr" in df_features and df_features["target_qqq_corr"].notna().any():
            log.info(f"  target_qqq_corr mean: {df_features['target_qqq_corr'].mean():.3f}  "
                     f"std: {df_features['target_qqq_corr'].std():.3f}")
            log.info(f"  target_smh_corr mean: {df_features['target_smh_corr'].mean():.3f}  "
                     f"std: {df_features['target_smh_corr'].std():.3f}")

        # Use new path structure: data/features/{SYMBOL}/features_w{window}.csv
        from ml.shared.paths import features_path, ensure_dirs
        ensure_dirs(symbol)
        out_path = args.out or features_path(symbol, w)
        df_features.to_csv(out_path, index=False)
        log.info(f"  Written to {out_path}")


if __name__ == "__main__":
    main()
