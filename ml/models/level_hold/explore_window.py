"""
explore_window.py

Determine optimal window size N and feature encoding for level_hold model.

Key constraint: find the SMALLEST N that produces reliable signal.
Larger window = later entry = worse entry quality.

Analyses:
1. Autocorrelation of bar features vs lag
2. Minimum window accuracy test (XGBoost with various N)
3. Encoding comparison (flattened per-bar vs window summaries)
4. Label distribution check (class balance, risk/reward)

Usage:
    python ml/models/level_hold/explore_window.py --symbols NVDA AMD
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from ml.shared.clickhouse import get_ch_client
from ml.shared.paths import features_path, REPORTS_DIR


# ---------------------------------------------------------------------------
# Model 1 Loading (from CSV + ClickHouse join)
# ---------------------------------------------------------------------------

def load_model1_predictions(symbol: str, window: int = 60) -> pd.DataFrame:
    """Load model 1 predictions from session_direction CSV."""
    path = features_path(symbol, window, prefix="session_direction")
    if not path.exists():
        log.error(f"Model 1 predictions not found: {path}")
        log.error(f"Run: python ml/models/session_direction/score_historical.py --symbols {symbol}")
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["session_date"])
    return df


def load_actual_labels(ch_client, symbol: str) -> pd.DataFrame:
    """Load actual directional labels from ClickHouse."""
    result = ch_client.query(LABELS_SQL, parameters={"symbol": symbol})
    df = pd.DataFrame(result.result_rows,
                      columns=["session_date", "actual_label", "fh_high_is_session_high", "fh_low_is_session_low"])
    df["session_date"] = pd.to_datetime(df["session_date"])
    return df


def get_correct_sessions(predictions: pd.DataFrame, actuals: pd.DataFrame) -> list[tuple]:
    """
    Join predictions with actuals and filter to sessions where model 1 was correct.

    Returns list of (session_date, direction_label) tuples.
    """
    predictions = predictions.copy()
    predictions["session_date"] = pd.to_datetime(predictions["session_date"])
    merged = predictions.merge(actuals, on="session_date", how="inner")

    # Model 1 is correct when:
    # - label=1 (fade_the_high) and fh_high_is_session_high=1
    # - label=0 (buy_the_dip) and fh_low_is_session_low=1
    correct_mask = (
        ((merged["label"] == 1) & (merged["fh_high_is_session_high"] == 1)) |
        ((merged["label"] == 0) & (merged["fh_low_is_session_low"] == 1))
    )
    correct = merged[correct_mask]

    return [(row["session_date"].date(), int(row["label"]))
            for _, row in correct.iterrows()]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL Queries
# ---------------------------------------------------------------------------

INTRADAY_BARS_SQL = """
SELECT ts, open, high, low, close, volume
FROM stock_bar_1m
WHERE symbol     = %(symbol)s
  AND session    = 1
  AND toDate(ts) = %(session_date)s
  AND (toHour(ts) > 10 OR (toHour(ts) = 10 AND toMinute(ts) >= 30))
ORDER BY ts
"""

CONTEXT_SQL = """
SELECT
    avg(session_range) AS atr20,
    avg(session_volume) AS avg_volume
FROM (
    SELECT
        toDate(ts)                    AS session_date,
        max(high) - min(low)          AS session_range,
        sum(volume)                   AS session_volume
    FROM stock_bar_1m
    WHERE symbol   = %(symbol)s
      AND session  = 1
      AND toDate(ts) < %(session_date)s
    GROUP BY session_date
    ORDER BY session_date DESC
    LIMIT 20
)
"""

# SQL to get actual labels for model 1 correctness filtering
LABELS_SQL = """
SELECT session_date, directional_label, fh_high_is_session_high, fh_low_is_session_low
FROM session_labels
WHERE symbol         = %(symbol)s
  AND window_minutes = 60
  AND directional_label IS NOT NULL
  AND session_date >= today() - INTERVAL 2 YEAR
ORDER BY session_date
"""


# ---------------------------------------------------------------------------
# Bar Feature Computation
# ---------------------------------------------------------------------------

def compute_bar_features(bars_df: pd.DataFrame, atr20: float, avg_volume: float) -> pd.DataFrame:
    """
    Compute per-bar features for a session.

    Features:
    - body_ratio: |close - open| / range
    - color: +1 if close > open, -1 if close < open, 0 if equal
    - upper_wick: (high - max(open, close)) / range
    - lower_wick: (min(open, close) - low) / range
    - close_position: (close - low) / range
    - volume_ratio: volume / avg_volume
    - range_ratio: bar_range / atr20
    """
    df = bars_df.copy()

    df["range"] = df["high"] - df["low"]
    df["range"] = df["range"].replace(0, np.nan)  # Avoid division by zero

    df["body"] = df["close"] - df["open"]
    df["body_ratio"] = df["body"].abs() / df["range"]

    df["color"] = np.sign(df["body"])

    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["range"]
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["range"]

    df["close_position"] = (df["close"] - df["low"]) / df["range"]

    df["volume_ratio"] = df["volume"] / avg_volume if avg_volume > 0 else 1.0
    df["range_ratio"] = df["range"] / atr20 if atr20 > 0 else 1.0

    # Fill NaN with neutral values
    df["body_ratio"] = df["body_ratio"].fillna(0.5)
    df["upper_wick"] = df["upper_wick"].fillna(0.0)
    df["lower_wick"] = df["lower_wick"].fillna(0.0)
    df["close_position"] = df["close_position"].fillna(0.5)

    return df


def compute_forward_return(bars_df: pd.DataFrame, idx: int, horizon: int) -> float:
    """Compute forward return over next horizon bars."""
    if idx + horizon >= len(bars_df):
        return np.nan
    current_price = bars_df.iloc[idx]["close"]
    future_price = bars_df.iloc[idx + horizon]["close"]
    return (future_price - current_price) / current_price


def compute_label(bars_df: pd.DataFrame, idx: int, direction_label: int, horizon: int) -> tuple[int, float]:
    """
    Compute entry label and risk/reward ratio.

    direction_label: 1 = fade_the_high (look for price to fall)
                     0 = buy_the_dip (look for price to rise)

    Returns: (good_entry: 0/1, risk_reward_ratio: float)
    """
    if idx + horizon >= len(bars_df):
        return np.nan, np.nan

    current_price = bars_df.iloc[idx]["close"]
    forward_bars = bars_df.iloc[idx + 1 : idx + horizon + 1]

    if direction_label == 1:  # fade_the_high -- look for price to fall
        max_favorable = current_price - forward_bars["low"].min()
        max_adverse = forward_bars["high"].max() - current_price
    else:  # buy_the_dip -- look for price to rise
        max_favorable = forward_bars["high"].max() - current_price
        max_adverse = current_price - forward_bars["low"].min()

    # Avoid division by zero
    if max_adverse < 0.01:
        rr = 3.0  # cap at 3.0 if no adverse move
    else:
        rr = max_favorable / max_adverse

    good_entry = 1 if rr >= 1.5 else 0
    return good_entry, rr


# ---------------------------------------------------------------------------
# Window Feature Encodings
# ---------------------------------------------------------------------------

def encode_flattened(bars_df: pd.DataFrame, idx: int, n_bars: int) -> Optional[dict]:
    """
    Encoding A: Flattened per-bar features.

    Features: bar1_body_ratio, bar2_body_ratio, ..., barN_body_ratio
              bar1_color, bar2_color, ..., barN_color
              etc.
    """
    if idx < n_bars - 1:
        return None

    features = {}
    feature_names = ["body_ratio", "color", "upper_wick", "lower_wick",
                     "close_position", "volume_ratio", "range_ratio"]

    for i in range(n_bars):
        bar_idx = idx - (n_bars - 1 - i)  # bar1 is oldest, barN is most recent
        bar = bars_df.iloc[bar_idx]
        for feat in feature_names:
            features[f"bar{i+1}_{feat}"] = bar[feat]

    return features


def encode_summarized(bars_df: pd.DataFrame, idx: int, n_bars: int) -> Optional[dict]:
    """
    Encoding B: Window-level summaries only.

    Features: momentum, momentum_change, color_streak, body_trend,
              range_trend, volume_trend, avg_body_ratio, avg_wick_ratio
    """
    if idx < n_bars - 1:
        return None

    window = bars_df.iloc[idx - n_bars + 1 : idx + 1]

    # Momentum: price change over window
    momentum = (window.iloc[-1]["close"] - window.iloc[0]["open"]) / window.iloc[0]["open"]

    # Momentum change: second half vs first half
    half = n_bars // 2
    if half > 0:
        first_half_momentum = (window.iloc[half]["close"] - window.iloc[0]["open"]) / window.iloc[0]["open"]
        second_half_momentum = (window.iloc[-1]["close"] - window.iloc[half]["open"]) / window.iloc[half]["open"]
        momentum_change = second_half_momentum - first_half_momentum
    else:
        momentum_change = 0.0

    # Color streak: consecutive bars of same color at end
    colors = window["color"].values
    streak = 1
    for i in range(len(colors) - 2, -1, -1):
        if colors[i] == colors[-1] and colors[-1] != 0:
            streak += 1
        else:
            break
    color_streak = streak * colors[-1]  # Signed streak

    # Body trend: regression slope of body_ratio
    body_ratios = window["body_ratio"].values
    x = np.arange(len(body_ratios))
    if len(body_ratios) > 1:
        body_trend = np.polyfit(x, body_ratios, 1)[0]
    else:
        body_trend = 0.0

    # Range trend: regression slope of range_ratio
    range_ratios = window["range_ratio"].values
    if len(range_ratios) > 1:
        range_trend = np.polyfit(x, range_ratios, 1)[0]
    else:
        range_trend = 0.0

    # Volume trend: regression slope of volume_ratio
    vol_ratios = window["volume_ratio"].values
    if len(vol_ratios) > 1:
        volume_trend = np.polyfit(x, vol_ratios, 1)[0]
    else:
        volume_trend = 0.0

    # Averages
    avg_body_ratio = window["body_ratio"].mean()
    avg_wick_ratio = (window["upper_wick"] + window["lower_wick"]).mean()

    return {
        "momentum": momentum,
        "momentum_change": momentum_change,
        "color_streak": color_streak,
        "body_trend": body_trend,
        "range_trend": range_trend,
        "volume_trend": volume_trend,
        "avg_body_ratio": avg_body_ratio,
        "avg_wick_ratio": avg_wick_ratio,
    }


# ---------------------------------------------------------------------------
# Analysis 1: Autocorrelation
# ---------------------------------------------------------------------------

def analyze_autocorrelation(ch_client, symbol: str, sessions: list[tuple[date, int]],
                            max_lag: int = 20) -> dict:
    """
    Compute autocorrelation of bar features and cross-correlation with forward return.
    """
    log.info("Analysis 1: Autocorrelation...")

    feature_names = ["body_ratio", "color", "upper_wick", "lower_wick",
                     "close_position", "volume_ratio", "range_ratio"]

    # Collect all bar features across sessions
    all_features = {f: [] for f in feature_names}
    all_forward_returns = []

    for session_date, direction_label in sessions[:200]:  # Limit for speed
        # Get context
        ctx_result = ch_client.query(
            CONTEXT_SQL,
            parameters={"symbol": symbol, "session_date": session_date.isoformat()}
        )
        if not ctx_result.result_rows:
            continue
        atr20, avg_volume = ctx_result.result_rows[0]
        if atr20 is None or avg_volume is None:
            continue

        # Get bars
        bars_result = ch_client.query(
            INTRADAY_BARS_SQL,
            parameters={"symbol": symbol, "session_date": session_date.isoformat()}
        )
        if not bars_result.result_rows:
            continue

        bars_df = pd.DataFrame(bars_result.result_rows, columns=["ts", "open", "high", "low", "close", "volume"])
        if len(bars_df) < max_lag + 20:
            continue

        bars_df = compute_bar_features(bars_df, atr20, avg_volume)

        # Collect features and forward returns
        for i in range(len(bars_df) - 15):
            for feat in feature_names:
                all_features[feat].append(bars_df.iloc[i][feat])
            fwd_ret = compute_forward_return(bars_df, i, 15)
            all_forward_returns.append(fwd_ret)

    # Convert to arrays
    for feat in feature_names:
        all_features[feat] = np.array(all_features[feat])
    all_forward_returns = np.array(all_forward_returns)

    # Compute autocorrelation for each feature
    autocorr = {f: [] for f in feature_names}
    for feat in feature_names:
        arr = all_features[feat]
        for lag in range(1, max_lag + 1):
            if len(arr) > lag:
                corr = np.corrcoef(arr[:-lag], arr[lag:])[0, 1]
                autocorr[feat].append(corr if not np.isnan(corr) else 0.0)
            else:
                autocorr[feat].append(0.0)

    # Compute cross-correlation with forward return
    crosscorr = {f: [] for f in feature_names}
    noise_threshold_lag = {}

    for feat in feature_names:
        arr = all_features[feat]
        fwd = all_forward_returns

        # Remove NaN from forward returns
        valid_mask = ~np.isnan(fwd)
        arr_valid = arr[valid_mask]
        fwd_valid = fwd[valid_mask]

        for lag in range(1, max_lag + 1):
            if len(arr_valid) > lag:
                # Feature at time t-lag vs forward return at time t
                corr = np.corrcoef(arr_valid[:-lag], fwd_valid[lag:])[0, 1]
                crosscorr[feat].append(corr if not np.isnan(corr) else 0.0)
            else:
                crosscorr[feat].append(0.0)

        # Find lag where cross-correlation drops below 0.05
        for i, c in enumerate(crosscorr[feat]):
            if abs(c) < 0.05:
                noise_threshold_lag[feat] = i + 1
                break
        else:
            noise_threshold_lag[feat] = max_lag  # Never dropped below threshold

    return {
        "autocorr": autocorr,
        "crosscorr": crosscorr,
        "noise_threshold_lag": noise_threshold_lag,
        "feature_names": feature_names,
        "max_lag": max_lag,
    }


# ---------------------------------------------------------------------------
# Analysis 2: Minimum Window Accuracy
# ---------------------------------------------------------------------------

def analyze_window_accuracy(ch_client, symbol: str, sessions: list[tuple[date, int]],
                            window_sizes: list[int] = [3, 5, 7, 10, 15]) -> dict:
    """
    Train XGBoost with different window sizes, measure accuracy.
    """
    log.info("Analysis 2: Window accuracy test...")

    results = {}

    for n_bars in window_sizes:
        log.info(f"  Testing N={n_bars}...")

        # Build dataset using summarized encoding
        X_all = []
        y_all = []

        for session_date, direction_label in sessions:
            # Get context
            ctx_result = ch_client.query(
                CONTEXT_SQL,
                parameters={"symbol": symbol, "session_date": session_date.isoformat()}
            )
            if not ctx_result.result_rows:
                continue
            atr20, avg_volume = ctx_result.result_rows[0]
            if atr20 is None or avg_volume is None:
                continue

            # Get bars
            bars_result = ch_client.query(
                INTRADAY_BARS_SQL,
                parameters={"symbol": symbol, "session_date": session_date.isoformat()}
            )
            if not bars_result.result_rows:
                continue

            bars_df = pd.DataFrame(bars_result.result_rows,
                                   columns=["ts", "open", "high", "low", "close", "volume"])
            if len(bars_df) < n_bars + 15:
                continue

            bars_df = compute_bar_features(bars_df, atr20, avg_volume)

            # Sample every 5th bar to avoid autocorrelation
            for i in range(n_bars - 1, len(bars_df) - 15, 5):
                features = encode_summarized(bars_df, i, n_bars)
                if features is None:
                    continue

                label, _ = compute_label(bars_df, i, direction_label, 15)
                if np.isnan(label):
                    continue

                X_all.append(features)
                y_all.append(label)

        if len(X_all) < 100:
            log.warning(f"  N={n_bars}: insufficient data ({len(X_all)} samples)")
            results[n_bars] = {"accuracy": np.nan, "auc": np.nan, "n_samples": len(X_all)}
            continue

        X_df = pd.DataFrame(X_all)
        y_arr = np.array(y_all)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        aucs = []

        for train_idx, test_idx in tscv.split(X_df):
            X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            accuracies.append(accuracy_score(y_test, y_pred))
            try:
                aucs.append(roc_auc_score(y_test, y_prob))
            except ValueError:
                aucs.append(0.5)  # Single class in test set

        results[n_bars] = {
            "accuracy": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "auc": np.mean(aucs),
            "auc_std": np.std(aucs),
            "n_samples": len(X_all),
        }
        log.info(f"  N={n_bars}: accuracy={results[n_bars]['accuracy']:.3f}, "
                 f"AUC={results[n_bars]['auc']:.3f}")

    # Find plateau (where accuracy stops improving significantly)
    valid_results = [(n, r) for n, r in results.items() if not np.isnan(r["accuracy"])]
    if valid_results:
        sorted_results = sorted(valid_results, key=lambda x: x[0])
        best_n = sorted_results[0][0]
        best_acc = sorted_results[0][1]["accuracy"]

        for n, r in sorted_results:
            if r["accuracy"] > best_acc + 0.01:  # 1% improvement threshold
                best_n = n
                best_acc = r["accuracy"]

        # Find minimum N within 1% of best
        for n, r in sorted_results:
            if r["accuracy"] >= best_acc - 0.01:
                plateau_n = n
                break
        else:
            plateau_n = best_n
    else:
        plateau_n = 5  # default

    return {
        "results": results,
        "plateau_n": plateau_n,
        "window_sizes": window_sizes,
    }


# ---------------------------------------------------------------------------
# Analysis 3: Encoding Comparison
# ---------------------------------------------------------------------------

def analyze_encoding_comparison(ch_client, symbol: str, sessions: list[tuple[date, int]],
                                 n_bars: int) -> dict:
    """
    Compare flattened per-bar vs summarized encoding.
    """
    log.info(f"Analysis 3: Encoding comparison (N={n_bars})...")

    results = {}

    for encoding_name, encode_fn in [("flattened", encode_flattened),
                                      ("summarized", encode_summarized)]:
        log.info(f"  Testing {encoding_name} encoding...")

        X_all = []
        y_all = []

        for session_date, direction_label in sessions:
            # Get context
            ctx_result = ch_client.query(
                CONTEXT_SQL,
                parameters={"symbol": symbol, "session_date": session_date.isoformat()}
            )
            if not ctx_result.result_rows:
                continue
            atr20, avg_volume = ctx_result.result_rows[0]
            if atr20 is None or avg_volume is None:
                continue

            # Get bars
            bars_result = ch_client.query(
                INTRADAY_BARS_SQL,
                parameters={"symbol": symbol, "session_date": session_date.isoformat()}
            )
            if not bars_result.result_rows:
                continue

            bars_df = pd.DataFrame(bars_result.result_rows,
                                   columns=["ts", "open", "high", "low", "close", "volume"])
            if len(bars_df) < n_bars + 15:
                continue

            bars_df = compute_bar_features(bars_df, atr20, avg_volume)

            # Sample every 5th bar
            for i in range(n_bars - 1, len(bars_df) - 15, 5):
                features = encode_fn(bars_df, i, n_bars)
                if features is None:
                    continue

                label, _ = compute_label(bars_df, i, direction_label, 15)
                if np.isnan(label):
                    continue

                X_all.append(features)
                y_all.append(label)

        if len(X_all) < 100:
            log.warning(f"  {encoding_name}: insufficient data")
            results[encoding_name] = {"accuracy": np.nan, "auc": np.nan}
            continue

        X_df = pd.DataFrame(X_all)
        y_arr = np.array(y_all)

        # Time-series CV
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        aucs = []
        feature_importances = np.zeros(len(X_df.columns))

        for train_idx, test_idx in tscv.split(X_df):
            X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            accuracies.append(accuracy_score(y_test, y_pred))
            try:
                aucs.append(roc_auc_score(y_test, y_prob))
            except ValueError:
                aucs.append(0.5)

            feature_importances += model.feature_importances_

        feature_importances /= 5  # Average across folds

        # For flattened encoding, find most important bar position
        most_important_bar = None
        if encoding_name == "flattened":
            bar_importance = {}
            for i, col in enumerate(X_df.columns):
                bar_num = int(col.split("_")[0].replace("bar", ""))
                bar_importance[bar_num] = bar_importance.get(bar_num, 0) + feature_importances[i]
            most_important_bar = max(bar_importance.items(), key=lambda x: x[1])[0]

        results[encoding_name] = {
            "accuracy": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "auc": np.mean(aucs),
            "auc_std": np.std(aucs),
            "n_features": len(X_df.columns),
            "feature_names": list(X_df.columns),
            "feature_importances": feature_importances,
            "most_important_bar": most_important_bar,
        }
        log.info(f"  {encoding_name}: accuracy={results[encoding_name]['accuracy']:.3f}, "
                 f"AUC={results[encoding_name]['auc']:.3f}, n_features={results[encoding_name]['n_features']}")

    # Determine recommended encoding
    flat_acc = results.get("flattened", {}).get("accuracy", 0) or 0
    summ_acc = results.get("summarized", {}).get("accuracy", 0) or 0

    if flat_acc > summ_acc + 0.02:  # 2% threshold for flattened to win
        recommended = "flattened"
    else:
        recommended = "summarized"  # Prefer simpler encoding if similar

    return {
        "results": results,
        "recommended": recommended,
        "n_bars": n_bars,
    }


# ---------------------------------------------------------------------------
# Analysis 4: Label Distribution
# ---------------------------------------------------------------------------

def analyze_label_distribution(ch_client, symbol: str, sessions: list[tuple[date, int]],
                                horizons: list[int] = [15, 30]) -> dict:
    """
    Check class balance and risk/reward distribution.
    """
    log.info("Analysis 4: Label distribution...")

    results = {}

    for horizon in horizons:
        log.info(f"  Horizon={horizon} bars...")

        labels = []
        rr_ratios = []
        time_of_day = []  # hour

        for session_date, direction_label in sessions:
            # Get bars
            bars_result = ch_client.query(
                INTRADAY_BARS_SQL,
                parameters={"symbol": symbol, "session_date": session_date.isoformat()}
            )
            if not bars_result.result_rows:
                continue

            bars_df = pd.DataFrame(bars_result.result_rows,
                                   columns=["ts", "open", "high", "low", "close", "volume"])
            if len(bars_df) < horizon + 5:
                continue

            # Sample every 10th bar for distribution analysis
            for i in range(0, len(bars_df) - horizon, 10):
                label, rr = compute_label(bars_df, i, direction_label, horizon)
                if np.isnan(label):
                    continue

                labels.append(label)
                rr_ratios.append(rr)

                ts = bars_df.iloc[i]["ts"]
                if isinstance(ts, datetime):
                    time_of_day.append(ts.hour + ts.minute / 60)
                else:
                    time_of_day.append(12)  # default

        if not labels:
            results[horizon] = {"good_pct": np.nan, "bad_pct": np.nan}
            continue

        labels = np.array(labels)
        rr_ratios = np.array(rr_ratios)
        time_of_day = np.array(time_of_day)

        good_pct = labels.mean() * 100
        bad_pct = 100 - good_pct

        # Class balance by time of day
        time_buckets = [10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5]
        balance_by_time = {}
        for t in time_buckets:
            mask = (time_of_day >= t) & (time_of_day < t + 0.5)
            if mask.sum() > 10:
                balance_by_time[t] = labels[mask].mean() * 100

        results[horizon] = {
            "good_pct": good_pct,
            "bad_pct": bad_pct,
            "rr_ratios": rr_ratios,
            "balance_by_time": balance_by_time,
            "n_samples": len(labels),
        }
        log.info(f"  Horizon={horizon}: {good_pct:.1f}% good, {bad_pct:.1f}% bad "
                 f"(n={len(labels)})")

    # Determine which horizon is more balanced
    h15_imbalance = abs(results.get(15, {}).get("good_pct", 50) - 50)
    h30_imbalance = abs(results.get(30, {}).get("good_pct", 50) - 50)

    more_balanced = 15 if h15_imbalance <= h30_imbalance else 30
    needs_weighting = h15_imbalance > 30 or h30_imbalance > 30  # >80/20 split

    return {
        "results": results,
        "horizons": horizons,
        "more_balanced": more_balanced,
        "needs_weighting": needs_weighting,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_autocorrelation(autocorr_results: dict, symbol: str, output_dir: Path):
    """Plot autocorrelation and cross-correlation vs lag."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    feature_names = autocorr_results["feature_names"]
    max_lag = autocorr_results["max_lag"]
    lags = range(1, max_lag + 1)

    # Autocorrelation
    ax = axes[0]
    for feat in feature_names:
        ax.plot(lags, autocorr_results["autocorr"][feat], label=feat, marker="o", markersize=3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Lag (bars)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"{symbol}: Feature Autocorrelation vs Lag")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Cross-correlation with forward return
    ax = axes[1]
    for feat in feature_names:
        ax.plot(lags, autocorr_results["crosscorr"][feat], label=feat, marker="o", markersize=3)
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="noise threshold")
    ax.axhline(y=-0.05, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Lag (bars)")
    ax.set_ylabel("Cross-correlation with forward return")
    ax.set_title(f"{symbol}: Predictive Signal Decay")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"{symbol}_autocorrelation.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved: {out_path}")


def plot_window_accuracy(window_results: dict, symbol: str, output_dir: Path):
    """Plot accuracy vs window size."""
    fig, ax = plt.subplots(figsize=(8, 5))

    window_sizes = window_results["window_sizes"]
    results = window_results["results"]

    accuracies = [results[n]["accuracy"] for n in window_sizes]
    aucs = [results[n]["auc"] for n in window_sizes]

    ax.plot(window_sizes, accuracies, "b-o", label="Accuracy", markersize=8)
    ax.plot(window_sizes, aucs, "g-s", label="AUC", markersize=8)

    ax.axvline(x=window_results["plateau_n"], color="red", linestyle="--",
               label=f"Plateau N={window_results['plateau_n']}")

    ax.set_xlabel("Window Size (bars)")
    ax.set_ylabel("Score")
    ax.set_title(f"{symbol}: Accuracy vs Window Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(window_sizes)

    plt.tight_layout()
    out_path = output_dir / f"{symbol}_accuracy_vs_window.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved: {out_path}")


def plot_encoding_comparison(encoding_results: dict, symbol: str, output_dir: Path):
    """Plot encoding comparison with feature importances."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    results = encoding_results["results"]

    # Bar chart of accuracy/AUC
    ax = axes[0]
    encodings = list(results.keys())
    x = np.arange(len(encodings))
    width = 0.35

    accuracies = [results[e]["accuracy"] for e in encodings]
    aucs = [results[e]["auc"] for e in encodings]

    ax.bar(x - width/2, accuracies, width, label="Accuracy", color="steelblue")
    ax.bar(x + width/2, aucs, width, label="AUC", color="seagreen")

    ax.set_ylabel("Score")
    ax.set_title(f"{symbol}: Encoding Comparison (N={encoding_results['n_bars']})")
    ax.set_xticks(x)
    ax.set_xticklabels(encodings)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Feature importances for best encoding
    ax = axes[1]
    best_encoding = encoding_results["recommended"]
    if best_encoding in results and results[best_encoding].get("feature_importances") is not None:
        importances = results[best_encoding]["feature_importances"]
        feature_names = results[best_encoding]["feature_names"]

        # Sort by importance
        sorted_idx = np.argsort(importances)[-15:]  # Top 15
        ax.barh(range(len(sorted_idx)),
                importances[sorted_idx],
                color="steelblue")
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top Features ({best_encoding})")

    plt.tight_layout()
    out_path = output_dir / f"{symbol}_encoding_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved: {out_path}")


def plot_label_distribution(label_results: dict, symbol: str, output_dir: Path):
    """Plot label distribution and risk/reward histogram."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    results = label_results["results"]
    horizons = label_results["horizons"]

    # Class balance pie charts
    for i, horizon in enumerate(horizons[:2]):
        ax = axes[0, i]
        if horizon in results and not np.isnan(results[horizon]["good_pct"]):
            good_pct = results[horizon]["good_pct"]
            bad_pct = results[horizon]["bad_pct"]
            ax.pie([good_pct, bad_pct],
                   labels=[f"Good ({good_pct:.1f}%)", f"Bad ({bad_pct:.1f}%)"],
                   colors=["seagreen", "salmon"],
                   autopct="%1.1f%%")
            ax.set_title(f"{symbol}: {horizon}-bar Horizon")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"{horizon}-bar Horizon")

    # Risk/reward histogram
    ax = axes[1, 0]
    for horizon in horizons[:2]:
        if horizon in results and "rr_ratios" in results[horizon]:
            rr = results[horizon]["rr_ratios"]
            rr_clipped = np.clip(rr, 0, 5)  # Clip for visualization
            ax.hist(rr_clipped, bins=50, alpha=0.5, label=f"{horizon}-bar")
    ax.axvline(x=1.5, color="red", linestyle="--", label="Good entry threshold")
    ax.set_xlabel("Risk/Reward Ratio")
    ax.set_ylabel("Count")
    ax.set_title(f"{symbol}: Risk/Reward Distribution")
    ax.legend()

    # Class balance by time of day
    ax = axes[1, 1]
    for horizon in horizons[:2]:
        if horizon in results and "balance_by_time" in results[horizon]:
            balance = results[horizon]["balance_by_time"]
            times = sorted(balance.keys())
            values = [balance[t] for t in times]
            ax.plot(times, values, "o-", label=f"{horizon}-bar")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time of Day (hour)")
    ax.set_ylabel("Good Entry %")
    ax.set_title(f"{symbol}: Class Balance by Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"{symbol}_label_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_symbol(symbol: str, ch_client) -> dict:
    """Run all analyses for a symbol."""
    log.info(f"Starting analysis for {symbol}")

    # Load model 1 predictions from CSV
    predictions = load_model1_predictions(symbol)
    if predictions.empty:
        return {}
    log.info(f"Loaded {len(predictions)} model 1 predictions")

    # Load actual labels from ClickHouse
    actuals = load_actual_labels(ch_client, symbol)
    if actuals.empty:
        log.error(f"No actual labels found for {symbol}")
        return {}
    log.info(f"Loaded {len(actuals)} actual labels")

    # Filter to correct sessions
    sessions = get_correct_sessions(predictions, actuals)
    if not sessions:
        log.error(f"No correct sessions found for {symbol}")
        return {}
    log.info(f"Found {len(sessions)} correct model 1 sessions (last 2 years)")

    # Run analyses
    autocorr_results = analyze_autocorrelation(ch_client, symbol, sessions)
    window_results = analyze_window_accuracy(ch_client, symbol, sessions)

    # Use plateau N for encoding comparison
    best_n = window_results["plateau_n"]
    encoding_results = analyze_encoding_comparison(ch_client, symbol, sessions, best_n)

    label_results = analyze_label_distribution(ch_client, symbol, sessions)

    # Generate plots
    output_dir = REPORTS_DIR / symbol
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating plots...")
    plot_autocorrelation(autocorr_results, symbol, output_dir)
    plot_window_accuracy(window_results, symbol, output_dir)
    plot_encoding_comparison(encoding_results, symbol, output_dir)
    plot_label_distribution(label_results, symbol, output_dir)

    return {
        "autocorr": autocorr_results,
        "window": window_results,
        "encoding": encoding_results,
        "label": label_results,
    }


def print_summary(symbol: str, results: dict):
    """Print formatted summary."""
    print(f"""
============================================================
WINDOW EXPLORATION SUMMARY: {symbol}
============================================================
""")

    # Autocorrelation
    if "autocorr" in results:
        ac = results["autocorr"]
        print("Autocorrelation analysis:")
        print("  Feature cross-correlations drop to noise at:")
        for feat, lag in ac["noise_threshold_lag"].items():
            print(f"    {feat:20s}: lag {lag}")
        suggested_n = min(ac["noise_threshold_lag"].values())
        print(f"  -> Suggested window from autocorrelation: N = {suggested_n} bars")

    # Window accuracy
    if "window" in results:
        wa = results["window"]
        print("\nAccuracy vs window size:")
        for n in wa["window_sizes"]:
            r = wa["results"][n]
            if not np.isnan(r["accuracy"]):
                print(f"  N={n:2d}: accuracy={r['accuracy']:.3f}, AUC={r['auc']:.3f}")
        print(f"  -> Accuracy plateaus at N = {wa['plateau_n']} bars")
        print(f"  -> Recommended N = {wa['plateau_n']} (minimum sufficient)")

    # Encoding comparison
    if "encoding" in results:
        ec = results["encoding"]
        print(f"\nEncoding comparison (N={ec['n_bars']}):")
        for enc, r in ec["results"].items():
            if not np.isnan(r["accuracy"]):
                print(f"  {enc:12s}: accuracy={r['accuracy']:.3f}, AUC={r['auc']:.3f}, "
                      f"n_features={r['n_features']}")
        print(f"  -> Recommended encoding: {ec['recommended']}")
        if ec["results"].get("flattened", {}).get("most_important_bar"):
            print(f"  -> Most important bar position in flattened: "
                  f"bar {ec['results']['flattened']['most_important_bar']} of {ec['n_bars']}")

    # Label distribution
    if "label" in results:
        ld = results["label"]
        print("\nLabel distribution:")
        for horizon in ld["horizons"]:
            r = ld["results"].get(horizon, {})
            if r and not np.isnan(r.get("good_pct", np.nan)):
                print(f"  {horizon:2d}-bar horizon: {r['good_pct']:.1f}% good entries, "
                      f"{r['bad_pct']:.1f}% bad entries")
        print(f"  -> {ld['more_balanced']}-bar is more balanced")
        print(f"  -> Class weighting {'needed' if ld['needs_weighting'] else 'not needed'}")

    # Final recommendations
    print("\nRECOMMENDED PARAMETERS:")
    if "window" in results:
        print(f"  Window size N:  {results['window']['plateau_n']} bars")
    if "encoding" in results:
        print(f"  Encoding:       {results['encoding']['recommended']}")
    if "label" in results:
        print(f"  Label horizon:  {results['label']['more_balanced']} bars")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Explore window size and encoding for level_hold model"
    )
    parser.add_argument("--symbols", nargs="+", required=True,
                        help="Symbols to analyze")
    args = parser.parse_args()

    ch_client = get_ch_client()

    all_results = {}
    for symbol in [s.upper() for s in args.symbols]:
        results = analyze_symbol(symbol, ch_client)
        if results:
            all_results[symbol] = results

    # Print summaries at the end
    for symbol, results in all_results.items():
        print_summary(symbol, results)

    log.info("Done.")


if __name__ == "__main__":
    main()
