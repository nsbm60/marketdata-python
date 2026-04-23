"""
ml/models/breakout/validate.py

Walk-forward validation of the breakout entry quality classifier.

Expanding window: train on all candidates up to each fold, test on next N.
Temporal ordering by ts. XGBoost binary classifier with balanced weights.

Usage:
    python -m ml.models.breakout.validate \
        --features data/features/breakout/features.csv \
        --fold-size 50

Outputs:
    data/reports/breakout/walkforward_report.txt
    data/reports/breakout/walkforward.png
"""

import argparse
from pathlib import Path

from ml.models.breakout.config import ModelConfig

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

FEATURES = [
    "symbol_encoded", "timeframe_encoded", "direction_encoded",
    "bar_index", "level_age_min", "ribbon_state_encoded",
    "ribbon_age", "ribbon_spread", "atr", "bar_range_atr",
    "bar_close_pct", "volume_ratio", "gap_pct", "score",
    "pivot_high_1_dist_atr", "pivot_high_1_age_bars",
    "pivot_high_2_dist_atr", "pivot_high_2_age_bars",
    "pivot_low_1_dist_atr", "pivot_low_1_age_bars",
    "pivot_low_2_dist_atr", "pivot_low_2_age_bars",
    "is_higher_high", "is_higher_low",
    "prior_session_high_dist_atr", "prior_session_low_dist_atr",
    "ribbon_spread_delta", "ribbon_spread_accel",
    "ema10_slope_3bar", "ema30_slope_3bar", "slope_differential",
    "was_compressed",
    "ema_ordered_count", "ribbon_upper_spread", "ribbon_lower_spread",
    "spread_ratio", "ema20_slope_3bar", "was_mixed_3bar", "is_resolving",
]
LABEL_COL = "label"

DEFAULT_MIN_TRAIN = 500
DEFAULT_FOLD_SIZE = 50
DEFAULT_CONFIDENCE = 0.65


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    df = df.dropna(subset=[LABEL_COL])
    return df


def train_fold(df_train: pd.DataFrame, features: list[str] = FEATURES) -> xgb.XGBClassifier:
    X = df_train[features]
    y = df_train[LABEL_COL]

    weights = compute_sample_weight(class_weight="balanced", y=y)

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

    # Internal validation for early stopping — last 15% of training data
    n_val = max(20, int(len(df_train) * 0.15))
    model.fit(
        X.iloc[:-n_val], y.iloc[:-n_val],
        sample_weight=weights[:-n_val],
        eval_set=[(X.iloc[-n_val:], y.iloc[-n_val:])],
        verbose=False,
    )
    return model


def run_walk_forward(
    df: pd.DataFrame,
    fold_size: int,
    min_train: int,
    confidence_threshold: float,
    features: list[str] = FEATURES,
) -> tuple[list[dict], pd.DataFrame]:
    """Expanding window walk-forward. Returns (fold results, predictions DataFrame)."""
    n = len(df)
    folds = []
    predictions = []
    start = min_train

    while start + fold_size <= n:
        df_train = df.iloc[:start]
        df_test = df.iloc[start:start + fold_size]

        # Skip folds where test set has only one class
        if df_test[LABEL_COL].nunique() < 2:
            start += fold_size
            continue

        model = train_fold(df_train, features)
        y_pred = model.predict(df_test[features])
        y_true = df_test[LABEL_COL].values
        acc = accuracy_score(y_true, y_pred)

        # Baseline: majority class in training set
        majority = df_train[LABEL_COL].mode()[0]
        baseline = (y_true == majority).mean()

        # Confidence: mean max probability
        probs = model.predict_proba(df_test[features])
        mean_conf = probs.max(axis=1).mean()

        # Positive class probability
        pos_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

        # High-confidence accuracy
        conf_mask = pos_probs >= confidence_threshold
        hc_acc = accuracy_score(y_true[conf_mask], y_pred[conf_mask]) \
            if conf_mask.sum() >= 5 else float("nan")
        hc_n = int(conf_mask.sum())

        # Collect per-candidate predictions
        fold_num = len(folds) + 1
        for i in range(len(df_test)):
            predictions.append({
                "symbol": df_test.iloc[i]["symbol"],
                "ts": df_test.iloc[i]["ts"],
                "timeframe": df_test.iloc[i]["timeframe"],
                "direction_encoded": int(df_test.iloc[i]["direction_encoded"]),
                "fold": fold_num,
                "prediction_prob": float(pos_probs[i]),
                "actual_label": int(y_true[i]),
                "is_hc": int(pos_probs[i] >= confidence_threshold),
            })

        fold = {
            "fold": fold_num,
            "train_start": df_train["ts"].min().date(),
            "train_end": df_train["ts"].max().date(),
            "test_start": df_test["ts"].min().date(),
            "test_end": df_test["ts"].max().date(),
            "train_n": len(df_train),
            "test_n": len(df_test),
            "accuracy": acc,
            "baseline": baseline,
            "lift": acc - baseline,
            "mean_conf": mean_conf,
            "hc_accuracy": hc_acc,
            "hc_n": hc_n,
        }
        folds.append(fold)
        start += fold_size

    return folds, pd.DataFrame(predictions)


def format_report(
    folds: list[dict],
    fold_size: int,
    min_train: int,
    confidence_threshold: float,
    mfe_threshold: float = 1.0,
    mae_threshold: float = 1.0,
) -> str:
    lines = []
    lines.append("=" * 75)
    lines.append("WALK-FORWARD VALIDATION REPORT -- Breakout Entry Quality")
    lines.append("=" * 75)
    lines.append(f"Fold size: {fold_size} candidates  |  Total folds: {len(folds)}")
    lines.append(f"Min training size: {min_train} candidates")
    lines.append(f"Label: mfe >= {mfe_threshold} ATR and mae <= {mae_threshold} ATR")
    lines.append("")

    # Per-fold table
    header = (f"{'Fold':>5}  {'Test Period':>23}  {'Train N':>7}  "
              f"{'Acc':>6}  {'Base':>6}  {'Lift':>6}  "
              f"{'HC Acc':>7}  {'HC N':>5}")
    lines.append(header)
    lines.append("-" * 75)

    for f in folds:
        period = f"{f['test_start']} -> {f['test_end']}"
        hc_acc = f"{f['hc_accuracy']:.1%}" if not np.isnan(f["hc_accuracy"]) else "  n/a"
        lines.append(
            f"{f['fold']:>5}  {period:>23}  {f['train_n']:>7}  "
            f"{f['accuracy']:>6.1%}  {f['baseline']:>6.1%}  {f['lift']:>+6.1%}  "
            f"{hc_acc:>7}  {f['hc_n']:>5}"
        )

    # Summary statistics
    accs = [f["accuracy"] for f in folds]
    lifts = [f["lift"] for f in folds]
    hc_accs = [f["hc_accuracy"] for f in folds if not np.isnan(f["hc_accuracy"])]
    pos_lifts = sum(1 for l in lifts if l > 0)

    lines.append("-" * 75)
    lines.append(f"\nSummary across {len(folds)} folds:")
    lines.append(f"  Mean accuracy:         {np.mean(accs):.1%}  "
                 f"(std: {np.std(accs):.1%})")
    lines.append(f"  Mean lift vs baseline: {np.mean(lifts):+.1%}  "
                 f"(std: {np.std(lifts):.1%})")
    lines.append(f"  Folds with +ve lift:   {pos_lifts}/{len(folds)}  "
                 f"({100 * pos_lifts / len(folds):.0f}%)")
    lines.append(f"  Min accuracy:          {min(accs):.1%}  "
                 f"(fold {accs.index(min(accs)) + 1})")
    lines.append(f"  Max accuracy:          {max(accs):.1%}  "
                 f"(fold {accs.index(max(accs)) + 1})")
    if hc_accs:
        lines.append(f"  Mean HC accuracy:      {np.mean(hc_accs):.1%}  "
                     f"(confidence >= {confidence_threshold})")
        total_hc = sum(f["hc_n"] for f in folds)
        hc_correct = sum(f["hc_n"] * f["hc_accuracy"] for f in folds if not np.isnan(f["hc_accuracy"]))
        overall_hc_acc = hc_correct / total_hc if total_hc > 0 else 0
        lines.append(f"  Overall HC accuracy:   {overall_hc_acc:.1%}  "
                     f"({int(hc_correct)}/{total_hc} correct across all folds)")

    return "\n".join(lines)


def plot_results(folds: list[dict], out_path: str, confidence_threshold: float):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    test_ends = [str(f["test_end"]) for f in folds]
    accs = [f["accuracy"] for f in folds]
    baselines = [f["baseline"] for f in folds]
    lifts = [f["lift"] for f in folds]
    hc_accs = [f["hc_accuracy"] for f in folds]

    x = range(len(folds))

    # Panel 1: accuracy vs baseline per fold
    axes[0].plot(x, accs, marker="o", label="Model accuracy", linewidth=1.5)
    axes[0].plot(x, baselines, marker="s", label="Baseline accuracy",
                 linewidth=1.5, linestyle="--", alpha=0.7)
    hc_valid = [(i, v) for i, v in enumerate(hc_accs) if not np.isnan(v)]
    if hc_valid:
        axes[0].plot([i for i, _ in hc_valid], [v for _, v in hc_valid],
                     marker="^", label=f"HC accuracy (>={confidence_threshold} conf)",
                     linewidth=1.5, linestyle=":", color="green")
    axes[0].axhline(np.mean(accs), color="blue", linestyle=":",
                    alpha=0.5, label=f"Mean acc={np.mean(accs):.1%}")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Breakout Entry Quality -- Walk-Forward Accuracy per Fold")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0.3, 1.0)

    # Panel 2: lift per fold
    colors = ["green" if l > 0 else "red" for l in lifts]
    axes[1].bar(x, lifts, color=colors, alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(np.mean(lifts), color="blue", linestyle=":",
                    alpha=0.7, label=f"Mean lift={np.mean(lifts):+.1%}")
    axes[1].set_ylabel("Lift vs baseline")
    axes[1].set_title("Lift over baseline per fold (green=positive, red=negative)")
    axes[1].legend(fontsize=8)

    # Panel 3: training set size over time
    train_ns = [f["train_n"] for f in folds]
    axes[2].plot(x, train_ns, color="gray", linewidth=1.5)
    axes[2].set_ylabel("Training candidates")
    axes[2].set_title("Training set size (expanding window)")

    # X axis labels
    tick_indices = list(range(0, len(folds), max(1, len(folds) // 12)))
    axes[2].set_xticks(tick_indices)
    axes[2].set_xticklabels([test_ends[i] for i in tick_indices],
                             rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward validation for breakout entry quality",
    )
    parser.add_argument("--features", required=True,
                        help="Features CSV path")
    parser.add_argument("--fold-size", type=int, default=DEFAULT_FOLD_SIZE,
                        help=f"Candidates per test fold (default: {DEFAULT_FOLD_SIZE})")
    parser.add_argument("--min-train", type=int, default=DEFAULT_MIN_TRAIN,
                        help=f"Min training candidates before first fold (default: {DEFAULT_MIN_TRAIN})")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE,
                        help=f"High-confidence threshold (default: {DEFAULT_CONFIDENCE})")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config YAML")
    parser.add_argument("--timeframe", type=str, default=None,
                        help="Single timeframe (removes timeframe_encoded from features)")
    args = parser.parse_args()

    # Load config if provided — CLI flags override config values
    cfg = ModelConfig.from_yaml(args.config) if args.config else None
    timeframe = args.timeframe or (cfg.timeframe if cfg else None)
    fold_size = args.fold_size if args.fold_size != DEFAULT_FOLD_SIZE else (cfg.fold_size if cfg else DEFAULT_FOLD_SIZE)
    min_train = args.min_train if args.min_train != DEFAULT_MIN_TRAIN else (cfg.min_train if cfg else DEFAULT_MIN_TRAIN)
    confidence = args.confidence_threshold if args.confidence_threshold != DEFAULT_CONFIDENCE else (cfg.confidence_threshold if cfg else DEFAULT_CONFIDENCE)
    mfe_threshold = cfg.mfe_threshold if cfg else 1.0
    mae_threshold = cfg.mae_threshold if cfg else 1.0

    # Adjust features for per-timeframe mode
    features = list(FEATURES)
    if timeframe:
        features = [f for f in features if f != "timeframe_encoded"]

    print(f"Loading {args.features}...")
    df = load(args.features)
    print(f"  {len(df)} candidates loaded "
          f"({df['ts'].min().date()} -> {df['ts'].max().date()})")

    pos = (df[LABEL_COL] == 1).sum()
    neg = (df[LABEL_COL] == 0).sum()
    print(f"  Label distribution: {pos} positive ({100 * pos / len(df):.1f}%), "
          f"{neg} negative ({100 * neg / len(df):.1f}%)")

    n_folds = (len(df) - min_train) // fold_size
    print(f"  Expected folds: ~{n_folds}  (fold_size={fold_size}, "
          f"min_train={min_train})")

    print("\nRunning walk-forward...")
    folds, predictions_df = run_walk_forward(df, fold_size, min_train, confidence, features)
    print(f"  Completed {len(folds)} folds")

    report = format_report(folds, fold_size, min_train, confidence,
                           mfe_threshold, mae_threshold)
    print("\n" + report)

    # Save report
    if timeframe:
        report_dir = Path(f"data/reports/breakout/{timeframe}")
    else:
        report_dir = Path("data/reports/breakout/combined")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_file = report_dir / "walkforward_report.txt"
    report_file.write_text(report)
    print(f"\nReport saved to {report_file}")

    # Save predictions
    if not predictions_df.empty:
        pred_path = report_dir / "predictions.csv"
        predictions_df.to_csv(pred_path, index=False)
        print(f"Predictions saved to {pred_path} ({len(predictions_df)} rows)")

    # Save plot
    plot_file = report_dir / "walkforward.png"
    plot_results(folds, str(plot_file), confidence)


if __name__ == "__main__":
    main()
