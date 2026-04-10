"""
explore_confidence.py

Determine whether model 1 session direction signal can be detected earlier
than the 60-minute window on high-confidence sessions.

Key question: On sessions where w60 confidence is high, was the signal
already present at w15, w30, or w45?

Usage:
    python ml/models/session_direction/explore_confidence.py --symbols NVDA AMD
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from ml.shared.features import FEATURES
from ml.shared.constants import DIRECTIONAL_LABEL_COL, DIRECTIONAL_NAMES, VALID_WINDOWS
from ml.shared.paths import (
    features_path, model_path, calibration_path, report_path, REPORTS_DIR, ensure_dirs
)
from ml.shared.clickhouse import get_ch_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL for actual labels
# ---------------------------------------------------------------------------

LABELS_SQL = """
SELECT session_date, directional_label
FROM session_labels
WHERE symbol         = %(symbol)s
  AND window_minutes = 60
  AND directional_label IS NOT NULL
ORDER BY session_date
"""


# ---------------------------------------------------------------------------
# Model training (same approach as train.py)
# ---------------------------------------------------------------------------

def load_features(symbol: str, window: int) -> pd.DataFrame:
    """Load feature matrix for a window. Requires all canonical FEATURES."""
    path = features_path(symbol, window)
    if not path.exists():
        log.warning(f"Feature matrix not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Require all canonical features
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"{symbol} w{window}: missing features: {missing}")

    # Fill nulls in correlation features
    corr_features = ["target_qqq_corr", "target_smh_corr", "target_qqq_beta"]
    for col in corr_features:
        if df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)

    # Keep only directional sessions (label 0 or 1)
    if DIRECTIONAL_LABEL_COL in df.columns:
        df = df[df[DIRECTIONAL_LABEL_COL].isin([0, 1])].copy()

    return df


def train_model(df: pd.DataFrame, window: int) -> tuple:
    """
    Train XGBoost model for directional prediction.
    Returns (model, calibrator) or (None, None) if insufficient data.
    """
    if len(df) < 100:
        log.warning(f"Insufficient data for w{window}: {len(df)} sessions")
        return None, None

    # Drop rows with missing features or labels
    df = df.dropna(subset=FEATURES + [DIRECTIONAL_LABEL_COL])

    if len(df) < 100:
        log.warning(f"Insufficient data after dropping nulls for w{window}: {len(df)}")
        return None, None

    X = df[FEATURES]
    y = df[DIRECTIONAL_LABEL_COL]

    # Use last 15% for validation/calibration
    n_val = max(20, int(len(df) * 0.15))
    X_train, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
    y_train, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

    weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        sample_weight=weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Calibrate
    raw_probs = model.predict_proba(X_val)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs[:, 1], y_val.values.astype(int))

    return model, calibrator


def get_or_train_model(symbol: str, window: int) -> tuple:
    """Load existing model or train if not found. Returns (model, calibrator)."""
    m_path = model_path(symbol, "session_direction", window)
    c_path = calibration_path(symbol, window)

    # Load features (validates all canonical FEATURES are present)
    df = load_features(symbol, window)
    if df.empty:
        return None, None

    if m_path.exists() and c_path.exists():
        log.info(f"Loading existing model for w{window}")
        model = xgb.XGBClassifier()
        model.load_model(str(m_path))
        with open(c_path, "rb") as f:
            calibrator = pickle.load(f)
        return model, calibrator

    log.info(f"Training model for w{window} with {len(FEATURES)} features...")
    model, calibrator = train_model(df, window)

    if model is not None:
        # Save for reuse
        ensure_dirs(symbol)
        model.save_model(str(m_path))
        with open(c_path, "wb") as f:
            pickle.dump(calibrator, f)
        log.info(f"Saved model to {m_path}")

    return model, calibrator


def apply_calibration(model, calibrator, X: pd.DataFrame) -> np.ndarray:
    """Get calibrated probabilities. Returns (n_samples, 2) array."""
    raw_probs = model.predict_proba(X)
    cal_p1 = calibrator.predict(raw_probs[:, 1])
    cal_p0 = 1 - cal_p1
    return np.column_stack([cal_p0, cal_p1])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_all_windows(symbol: str, windows: list[int]) -> pd.DataFrame:
    """
    Score all sessions at each window, return joined DataFrame.

    Columns: session_date, w15_pred, w15_conf, w30_pred, w30_conf, etc.
    """
    all_scores = {}

    for window in windows:
        model, calibrator = get_or_train_model(symbol, window)
        if model is None:
            log.warning(f"No model for w{window}")
            continue

        # Reload features for scoring
        df = load_features(symbol, window)
        if df.empty:
            continue

        # Score all sessions using canonical FEATURES
        X = df[FEATURES]
        cal_probs = apply_calibration(model, calibrator, X)

        scores = pd.DataFrame({
            "session_date": df["date"].dt.date,
            f"w{window}_pred": cal_probs.argmax(axis=1),
            f"w{window}_conf": cal_probs.max(axis=1),
            f"w{window}_close_position": df["close_position"].values,
        })
        all_scores[window] = scores

    if not all_scores:
        return pd.DataFrame()

    # Join all windows on session_date
    result = None
    for window, scores in all_scores.items():
        if result is None:
            result = scores
        else:
            result = result.merge(scores, on="session_date", how="inner")

    return result


def add_actual_labels(scores_df: pd.DataFrame, ch_client, symbol: str) -> pd.DataFrame:
    """Add actual directional labels from ClickHouse."""
    result = ch_client.query(LABELS_SQL, parameters={"symbol": symbol})
    if not result.result_rows:
        return scores_df

    labels = pd.DataFrame(result.result_rows, columns=["session_date", "actual_label"])
    labels["session_date"] = pd.to_datetime(labels["session_date"]).dt.date

    return scores_df.merge(labels, on="session_date", how="inner")


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_entry_quality(close_position: float, label: int) -> float:
    """
    Compute entry quality based on close_position and direction.
    label=1 (fade_the_high): close_position = opportunity remaining
    label=0 (buy_the_dip): 1 - close_position = opportunity remaining
    """
    if label == 1:
        return close_position
    else:
        return 1.0 - close_position


def analyze_confidence_progression(df: pd.DataFrame, windows: list[int]) -> dict:
    """Analysis 1: How does confidence evolve across windows?"""
    results = {"windows": windows}

    for label in [0, 1]:
        label_name = DIRECTIONAL_NAMES[label]
        subset = df[df["w60_pred"] == label]

        avg_conf = []
        for w in windows:
            col = f"w{w}_conf"
            if col in df.columns:
                avg_conf.append(subset[col].mean())
            else:
                avg_conf.append(np.nan)
        results[f"{label_name}_avg_conf"] = avg_conf

    # Distribution of earlier window confidence for high-confidence w60 sessions
    high_conf = df[df["w60_conf"] >= 0.70]
    for w in [30, 45]:
        col = f"w{w}_conf"
        if col in df.columns:
            results[f"w{w}_conf_when_w60_high"] = high_conf[col].values

    return results


def analyze_agreement_rate(df: pd.DataFrame, windows: list[int]) -> dict:
    """Analysis 2: Agreement rate between earlier windows and w60."""
    results = []

    for w in windows:
        pred_col = f"w{w}_pred"
        conf_col = f"w{w}_conf"

        if pred_col not in df.columns:
            continue

        agrees = df[pred_col] == df["w60_pred"]
        agreement_rate = agrees.mean()
        avg_conf_when_agreeing = df.loc[agrees, conf_col].mean()

        results.append({
            "window": w,
            "agreement_rate": agreement_rate,
            "avg_conf_when_agreeing": avg_conf_when_agreeing,
            "n_sessions": len(df),
        })

    return results


def analyze_early_signal_accuracy(df: pd.DataFrame, windows: list[int]) -> dict:
    """Analysis 3: When wN agrees with w60, is it actually correct?"""
    results = []

    for w in windows:
        pred_col = f"w{w}_pred"
        if pred_col not in df.columns:
            continue

        agrees = df[pred_col] == df["w60_pred"]
        agreeing_sessions = df[agrees]

        if len(agreeing_sessions) == 0:
            continue

        accuracy = (agreeing_sessions[pred_col] == agreeing_sessions["actual_label"]).mean()

        results.append({
            "window": w,
            "n_sessions": len(agreeing_sessions),
            "accuracy": accuracy,
        })

    # Add w60 baseline
    w60_accuracy = (df["w60_pred"] == df["actual_label"]).mean()
    for r in results:
        r["vs_w60_baseline"] = r["accuracy"] - w60_accuracy

    results.append({
        "window": 60,
        "n_sessions": len(df),
        "accuracy": w60_accuracy,
        "vs_w60_baseline": 0.0,
    })

    return results


def analyze_entry_quality_benefit(df: pd.DataFrame, windows: list[int]) -> dict:
    """Analysis 4: Entry quality improvement from firing earlier."""
    results = []

    # Compute entry quality at w60 for correct sessions
    df = df.copy()
    df["w60_entry_quality"] = df.apply(
        lambda r: compute_entry_quality(r["w60_close_position"], int(r["w60_pred"])),
        axis=1
    )
    w60_baseline = df["w60_entry_quality"].mean()

    for w in windows:
        pred_col = f"w{w}_pred"
        cp_col = f"w{w}_close_position"

        if pred_col not in df.columns or cp_col not in df.columns:
            continue

        agrees = df[pred_col] == df["w60_pred"]
        agreeing = df[agrees].copy()

        if len(agreeing) == 0:
            continue

        agreeing[f"w{w}_entry_quality"] = agreeing.apply(
            lambda r: compute_entry_quality(r[cp_col], int(r[pred_col])),
            axis=1
        )

        avg_eq = agreeing[f"w{w}_entry_quality"].mean()
        improvement = avg_eq - w60_baseline

        results.append({
            "window": w,
            "avg_entry_quality": avg_eq,
            "improvement_vs_w60": improvement,
            "n_sessions": len(agreeing),
        })

    results.append({
        "window": 60,
        "avg_entry_quality": w60_baseline,
        "improvement_vs_w60": 0.0,
        "n_sessions": len(df),
    })

    return results


def analyze_confidence_thresholds(df: pd.DataFrame, windows: list[int],
                                   thresholds: list[float] = [0.60, 0.65, 0.70, 0.75, 0.80]) -> dict:
    """Analysis 5: Optimal confidence threshold at each window."""
    results = {}

    # Compute entry quality at each window
    df = df.copy()
    for w in windows:
        cp_col = f"w{w}_close_position"
        pred_col = f"w{w}_pred"
        if cp_col in df.columns and pred_col in df.columns:
            df[f"w{w}_entry_quality"] = df.apply(
                lambda r: compute_entry_quality(r[cp_col], int(r[pred_col])),
                axis=1
            )

    for w in windows:
        conf_col = f"w{w}_conf"
        pred_col = f"w{w}_pred"
        eq_col = f"w{w}_entry_quality"

        if conf_col not in df.columns:
            continue

        window_results = []
        for thresh in thresholds:
            mask = df[conf_col] >= thresh
            subset = df[mask]

            if len(subset) == 0:
                continue

            coverage = len(subset) / len(df)
            accuracy = (subset[pred_col] == subset["actual_label"]).mean()

            avg_eq = subset[eq_col].mean() if eq_col in df.columns else np.nan

            window_results.append({
                "threshold": thresh,
                "coverage": coverage,
                "accuracy": accuracy,
                "avg_entry_quality": avg_eq,
                "n_sessions": len(subset),
            })

        results[w] = window_results

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confidence_progression(results: dict, symbol: str, output_dir: Path):
    """Plot average confidence across windows."""
    fig, ax = plt.subplots(figsize=(10, 6))

    windows = results["windows"]

    for label in [0, 1]:
        label_name = DIRECTIONAL_NAMES[label]
        key = f"{label_name}_avg_conf"
        if key in results:
            ax.plot(windows, results[key], "o-", label=label_name, markersize=8)

    ax.set_xlabel("Window (minutes)")
    ax.set_ylabel("Average Confidence")
    ax.set_title(f"{symbol}: Confidence Progression Across Windows")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(windows)

    plt.tight_layout()
    out_path = output_dir / f"{symbol}_confidence_progression.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved: {out_path}")


def plot_accuracy_vs_coverage(threshold_results: dict, symbol: str, output_dir: Path):
    """Plot accuracy vs coverage tradeoff for each window."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {15: "red", 30: "orange", 45: "green", 60: "blue"}

    for w, results in threshold_results.items():
        if not results:
            continue
        coverages = [r["coverage"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        ax.plot(coverages, accuracies, "o-", label=f"w{w}",
                color=colors.get(w, "gray"), markersize=8)

        # Annotate with thresholds
        for r in results:
            ax.annotate(f"{r['threshold']:.2f}",
                        (r["coverage"], r["accuracy"]),
                        textcoords="offset points",
                        xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Coverage (% of sessions)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{symbol}: Accuracy vs Coverage by Window")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"{symbol}_accuracy_vs_coverage.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved: {out_path}")


def plot_entry_quality_scatter(df: pd.DataFrame, symbol: str, output_dir: Path):
    """Scatter plot: w60 confidence vs entry quality improvement at w30."""
    if "w30_close_position" not in df.columns:
        return

    df = df.copy()
    df["w60_entry_quality"] = df.apply(
        lambda r: compute_entry_quality(r["w60_close_position"], int(r["w60_pred"])),
        axis=1
    )
    df["w30_entry_quality"] = df.apply(
        lambda r: compute_entry_quality(r["w30_close_position"], int(r["w30_pred"])),
        axis=1
    )
    df["improvement"] = df["w30_entry_quality"] - df["w60_entry_quality"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df["w60_conf"], df["improvement"], alpha=0.5, s=20)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("w60 Confidence")
    ax.set_ylabel("Entry Quality Improvement (w30 - w60)")
    ax.set_title(f"{symbol}: Entry Quality Benefit vs w60 Confidence")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"{symbol}_entry_quality_scatter.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_symbol(symbol: str, ch_client) -> dict:
    """Run all analyses for a symbol."""
    log.info(f"Starting analysis for {symbol}")

    # Determine available windows
    available_windows = []
    for w in VALID_WINDOWS:
        if features_path(symbol, w).exists():
            available_windows.append(w)
    log.info(f"Available windows: {available_windows}")

    if 60 not in available_windows:
        log.error(f"w60 features required but not found for {symbol}")
        return {}

    # Score all sessions at each window
    scores_df = score_all_windows(symbol, available_windows)
    if scores_df.empty:
        log.error(f"No scores computed for {symbol}")
        return {}
    log.info(f"Scored {len(scores_df)} sessions across windows")

    # Add actual labels
    scores_df = add_actual_labels(scores_df, ch_client, symbol)
    log.info(f"After joining labels: {len(scores_df)} sessions")

    # Filter to correct w60 predictions
    correct_w60 = scores_df[scores_df["w60_pred"] == scores_df["actual_label"]].copy()
    log.info(f"Correct w60 sessions: {len(correct_w60)} ({100*len(correct_w60)/len(scores_df):.1f}%)")

    # Run analyses on correct w60 sessions
    log.info("Running analyses...")

    conf_progression = analyze_confidence_progression(correct_w60, available_windows)
    agreement_rate = analyze_agreement_rate(correct_w60, available_windows)
    early_accuracy = analyze_early_signal_accuracy(correct_w60, available_windows)
    entry_quality = analyze_entry_quality_benefit(correct_w60, available_windows)
    threshold_results = analyze_confidence_thresholds(correct_w60, available_windows)

    # Generate plots
    output_dir = REPORTS_DIR / symbol
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating plots...")
    plot_confidence_progression(conf_progression, symbol, output_dir)
    plot_accuracy_vs_coverage(threshold_results, symbol, output_dir)
    plot_entry_quality_scatter(correct_w60, symbol, output_dir)

    return {
        "available_windows": available_windows,
        "n_sessions": len(scores_df),
        "n_correct_w60": len(correct_w60),
        "w60_accuracy": len(correct_w60) / len(scores_df),
        "conf_progression": conf_progression,
        "agreement_rate": agreement_rate,
        "early_accuracy": early_accuracy,
        "entry_quality": entry_quality,
        "threshold_results": threshold_results,
    }


def find_best_early_fire(results: dict) -> dict:
    """Find the best early-fire configuration."""
    threshold_results = results.get("threshold_results", {})
    w60_accuracy = results.get("w60_accuracy", 0.78)

    best = None
    for w in [15, 30, 45]:
        if w not in threshold_results:
            continue

        for r in threshold_results[w]:
            # Requirements: accuracy >= w60 baseline - 3%, coverage >= 30%
            if r["accuracy"] >= w60_accuracy - 0.03 and r["coverage"] >= 0.30:
                if best is None or r["avg_entry_quality"] > best["avg_entry_quality"]:
                    best = {
                        "window": w,
                        "threshold": r["threshold"],
                        "coverage": r["coverage"],
                        "accuracy": r["accuracy"],
                        "avg_entry_quality": r["avg_entry_quality"],
                    }

    return best


def print_summary(symbol: str, results: dict):
    """Print formatted summary."""
    if not results:
        print(f"\n{symbol}: No results\n")
        return

    print(f"""
============================================================
CONFIDENCE EXPLORATION SUMMARY: {symbol}
============================================================
""")

    # Overall w60 accuracy
    w60_acc = results.get("w60_accuracy", 0)
    n_sessions = results.get("n_sessions", 0)
    n_correct = results.get("n_correct_w60", 0)
    print(f"Sessions analyzed: {n_sessions}")
    print(f"w60 accuracy: {w60_acc:.1%} ({n_correct} correct sessions)")

    # Agreement with w60 (on correct w60 sessions)
    print("\nAgreement with w60 prediction (on correct w60 sessions):")
    agreement = results.get("agreement_rate", [])
    if agreement:
        line = "  "
        for r in agreement:
            if r["window"] != 60:
                line += f"w{r['window']}: {r['agreement_rate']:.1%}  "
        print(line if line.strip() else "  (no earlier windows available)")
    else:
        print("  (no earlier windows available)")

    # Entry quality improvement
    print("\nEntry quality improvement vs w60 baseline:")
    eq_results = results.get("entry_quality", [])
    w60_baseline = next((r["avg_entry_quality"] for r in eq_results if r["window"] == 60), 0.241)
    line = f"  (w60 baseline: {w60_baseline:.3f})\n  "
    for r in eq_results:
        if r["window"] != 60:
            line += f"w{r['window']}: {r['improvement_vs_w60']:+.3f}  "
    print(line)

    # Best early-fire configuration
    best = find_best_early_fire(results)
    print("\nBest early-fire configuration:")
    if best:
        print(f"  Window:            w{best['window']}")
        print(f"  Threshold:         {best['threshold']:.2f}")
        print(f"  Coverage:          {best['coverage']:.1%} of sessions")
        print(f"  Accuracy:          {best['accuracy']:.1%}")
        print(f"  Avg entry_quality: {best['avg_entry_quality']:.3f} (vs {w60_baseline:.3f} at w60)")
    else:
        print("  No early-fire configuration meets requirements")

    # Recommendation
    print("\nRECOMMENDATION:")
    if best:
        improvement = best["avg_entry_quality"] - w60_baseline
        if improvement > 0.05:
            print(f"  Fire at w{best['window']} when confidence > {best['threshold']:.2f}, "
                  f"otherwise wait for w60")
            print(f"  Expected entry quality improvement: {improvement:+.3f}")
        else:
            print("  Early firing not justified -- entry quality improvement too small")
    else:
        print("  No early firing justified -- accuracy drop too large")

    print("=" * 60)


def generate_symbol_config(symbol: str, results: dict) -> dict:
    """Generate config recommendation for a symbol based on analysis results."""
    best = find_best_early_fire(results)
    w60_baseline = 0.241  # Default baseline

    # Get w60 entry quality from results
    eq_results = results.get("entry_quality", [])
    for r in eq_results:
        if r["window"] == 60:
            w60_baseline = r["avg_entry_quality"]
            break

    if best and best["avg_entry_quality"] - w60_baseline > 0.05:
        # Early fire justified
        return {
            "window_minutes": best["window"],
            "confidence_threshold": best["threshold"],
            "fallback_window": 60,
            "entry_quality_improvement": round(best["avg_entry_quality"] - w60_baseline, 3),
        }
    else:
        # Standard w60 fire
        return {
            "window_minutes": 60,
            "confidence_threshold": 0.65,
            "fallback_window": None,
            "entry_quality_improvement": 0.0,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Explore early confidence detection for model 1"
    )
    parser.add_argument("--symbols", nargs="+", required=True,
                        help="Symbols to analyze")
    parser.add_argument("--update-config", action="store_true",
                        help="Update prediction_windows.yaml with recommendations")
    args = parser.parse_args()

    ch_client = get_ch_client()

    all_results = {}
    for symbol in [s.upper() for s in args.symbols]:
        results = analyze_symbol(symbol, ch_client)
        if results:
            all_results[symbol] = results

    # Print summaries
    for symbol, results in all_results.items():
        print_summary(symbol, results)

    # Update config if requested
    if args.update_config:
        from ml.shared.config import update_symbol_config

        log.info("Updating prediction_windows.yaml...")
        for symbol, results in all_results.items():
            cfg = generate_symbol_config(symbol, results)
            update_symbol_config(
                symbol=symbol,
                window_minutes=cfg["window_minutes"],
                confidence_threshold=cfg["confidence_threshold"],
                fallback_window=cfg["fallback_window"],
                entry_quality_improvement=cfg["entry_quality_improvement"],
            )
            log.info(f"  {symbol}: window={cfg['window_minutes']} "
                     f"threshold={cfg['confidence_threshold']} "
                     f"fallback={cfg['fallback_window']}")
        log.info("Config updated.")

    log.info("Done.")


if __name__ == "__main__":
    main()
