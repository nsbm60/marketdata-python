"""
walk_forward.py

Walk-forward validation of the directional session classifier.

Slides a test window forward through time, training on all sessions
up to each fold and testing on the next N sessions. Produces a fold-by-fold
accuracy report and plots showing model consistency over time.

Usage:
    python walk_forward.py --features nvda_features_w60.csv
    python walk_forward.py --features nvda_features_w60.csv --fold-size 20
    python walk_forward.py --features nvda_features_w60.csv --mode directional

Outputs:
    <symbol>_walkforward_report.txt
    <symbol>_walkforward.png

Dependencies:
    pip install xgboost scikit-learn pandas numpy matplotlib
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Must match train_model.py exactly
# ---------------------------------------------------------------------------

FEATURES = [
    "gap_pct", "prior_day_range_pct", "atr20",
    "w_range_pct", "w_range_atr", "w_vwap_dev",
    "f15_range_ratio", "f15_vol_ratio",
    "w_vol_ratio",
    "sweep_signal", "sweep_direction",
    "reversal_progress", "close_position",
    "rolling_reversal_rate", "rolling_high_set_rate",
    "directional_bias", "gap_regime_alignment",
    "nvda_qqq_corr", "nvda_smh_corr", "nvda_qqq_beta",
]

LABEL_COLS = {
    "directional": "directional_label",
    "binary":      "binary_label",
    "multiclass":  "label",
}

LABEL_NAMES = {
    "directional": {0: "buy_the_dip", 1: "fade_the_high"},
    "binary":      {0: "non_reversal", 1: "reversal"},
    "multiclass":  {0: "trend", 1: "containment", 2: "reversal", 3: "double_sweep"},
}

# Minimum training sessions before we attempt the first fold
MIN_TRAIN = 200


def load(path: str, label_col: str, directional: bool) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Fill correlation feature nulls with median
    for col in ["nvda_qqq_corr", "nvda_smh_corr", "nvda_qqq_beta"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=FEATURES + [label_col])

    if directional:
        df = df[df[label_col].isin([0, 1])].copy()

    return df


def train_fold(df_train: pd.DataFrame, label_col: str, binary: bool) -> xgb.XGBClassifier:
    X = df_train[FEATURES]
    y = df_train[label_col]

    weights  = compute_sample_weight(class_weight="balanced", y=y)
    obj      = "binary:logistic" if binary else "multi:softprob"
    metric   = "logloss"         if binary else "mlogloss"
    extra    = {}                 if binary else {"num_class": 4}

    model = xgb.XGBClassifier(
        objective        = obj,
        eval_metric      = metric,
        **extra,
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        early_stopping_rounds = 30,
        random_state     = 42,
        verbosity        = 0,
    )

    # Internal validation for early stopping -- last 15% of training data
    n_val = max(20, int(len(df_train) * 0.15))
    model.fit(
        X.iloc[:-n_val], y.iloc[:-n_val],
        sample_weight = weights[:-n_val],
        eval_set      = [(X.iloc[-n_val:], y.iloc[-n_val:])],
        verbose       = False,
    )
    return model


def run_walk_forward(df: pd.DataFrame,
                     label_col: str,
                     fold_size: int,
                     binary: bool) -> list[dict]:
    """
    Expanding window walk-forward.
    Returns list of fold result dicts.
    """
    n       = len(df)
    folds   = []
    start   = MIN_TRAIN

    while start + fold_size <= n:
        df_train = df.iloc[:start]
        df_test  = df.iloc[start:start + fold_size]

        # Skip folds where test set has only one class (too small to evaluate)
        if df_test[label_col].nunique() < 2:
            start += fold_size
            continue

        model   = train_fold(df_train, label_col, binary)
        y_pred  = model.predict(df_test[FEATURES])
        y_true  = df_test[label_col].values
        acc     = accuracy_score(y_true, y_pred)

        # Baseline: majority class in training set
        majority    = df_train[label_col].mode()[0]
        baseline    = (y_true == majority).mean()

        # Confidence: mean max probability
        probs      = model.predict_proba(df_test[FEATURES])
        mean_conf  = probs.max(axis=1).mean()

        # High-confidence accuracy (>= 0.65)
        conf_mask  = probs.max(axis=1) >= 0.65
        hc_acc     = accuracy_score(y_true[conf_mask], y_pred[conf_mask]) \
                     if conf_mask.sum() >= 5 else float("nan")
        hc_n       = conf_mask.sum()

        fold = {
            "fold":          len(folds) + 1,
            "train_start":   df_train["date"].min().date(),
            "train_end":     df_train["date"].max().date(),
            "test_start":    df_test["date"].min().date(),
            "test_end":      df_test["date"].max().date(),
            "train_n":       len(df_train),
            "test_n":        len(df_test),
            "accuracy":      acc,
            "baseline":      baseline,
            "lift":          acc - baseline,
            "mean_conf":     mean_conf,
            "hc_accuracy":   hc_acc,
            "hc_n":          hc_n,
        }
        folds.append(fold)
        start += fold_size

    return folds


def format_report(folds: list[dict], symbol: str, mode: str,
                  fold_size: int) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"WALK-FORWARD VALIDATION REPORT -- {symbol} ({mode})")
    lines.append("=" * 70)
    lines.append(f"Fold size: {fold_size} sessions  |  Total folds: {len(folds)}")
    lines.append(f"Min training size: {MIN_TRAIN} sessions")
    lines.append("")

    # Per-fold table
    header = (f"{'Fold':>5}  {'Test Period':>23}  {'Train N':>7}  "
              f"{'Acc':>6}  {'Base':>6}  {'Lift':>6}  "
              f"{'HC Acc':>7}  {'HC N':>5}")
    lines.append(header)
    lines.append("-" * 75)

    for f in folds:
        period = f"{f['test_start']} → {f['test_end']}"
        hc_acc = f"{f['hc_accuracy']:.1%}" if not np.isnan(f['hc_accuracy']) else "  n/a"
        lines.append(
            f"{f['fold']:>5}  {period:>23}  {f['train_n']:>7}  "
            f"{f['accuracy']:>6.1%}  {f['baseline']:>6.1%}  {f['lift']:>+6.1%}  "
            f"{hc_acc:>7}  {f['hc_n']:>5}"
        )

    # Summary statistics
    accs      = [f["accuracy"]   for f in folds]
    lifts     = [f["lift"]       for f in folds]
    hc_accs   = [f["hc_accuracy"] for f in folds if not np.isnan(f["hc_accuracy"])]
    pos_lifts = sum(1 for l in lifts if l > 0)

    lines.append("-" * 75)
    lines.append(f"\nSummary across {len(folds)} folds:")
    lines.append(f"  Mean accuracy:         {np.mean(accs):.1%}  "
                 f"(std: {np.std(accs):.1%})")
    lines.append(f"  Mean lift vs baseline: {np.mean(lifts):+.1%}  "
                 f"(std: {np.std(lifts):.1%})")
    lines.append(f"  Folds with +ve lift:   {pos_lifts}/{len(folds)}  "
                 f"({100*pos_lifts/len(folds):.0f}%)")
    lines.append(f"  Min accuracy:          {min(accs):.1%}  "
                 f"(fold {accs.index(min(accs))+1})")
    lines.append(f"  Max accuracy:          {max(accs):.1%}  "
                 f"(fold {accs.index(max(accs))+1})")
    if hc_accs:
        lines.append(f"  Mean HC accuracy:      {np.mean(hc_accs):.1%}  "
                     f"(confidence >= 0.65)")

    return "\n".join(lines)


def plot_results(folds: list[dict], symbol: str, mode: str, out_path: str):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    test_ends = [str(f["test_end"]) for f in folds]
    accs      = [f["accuracy"]    for f in folds]
    baselines = [f["baseline"]    for f in folds]
    lifts     = [f["lift"]        for f in folds]
    hc_accs   = [f["hc_accuracy"] for f in folds]

    x = range(len(folds))

    # Panel 1: accuracy vs baseline per fold
    axes[0].plot(x, accs,      marker="o", label="Model accuracy",    linewidth=1.5)
    axes[0].plot(x, baselines, marker="s", label="Baseline accuracy",
                 linewidth=1.5, linestyle="--", alpha=0.7)
    hc_valid = [(i, v) for i, v in enumerate(hc_accs) if not np.isnan(v)]
    if hc_valid:
        axes[0].plot([i for i, _ in hc_valid], [v for _, v in hc_valid],
                     marker="^", label="HC accuracy (≥0.65 conf)",
                     linewidth=1.5, linestyle=":", color="green")
    axes[0].axhline(np.mean(accs), color="blue", linestyle=":",
                    alpha=0.5, label=f"Mean acc={np.mean(accs):.1%}")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{symbol} Walk-Forward Validation ({mode}) -- Accuracy per Fold")
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
    axes[2].set_ylabel("Training sessions")
    axes[2].set_title("Training set size (expanding window)")

    # X axis labels -- show every 3rd fold to avoid crowding
    tick_indices = list(range(0, len(folds), max(1, len(folds) // 12)))
    axes[2].set_xticks(tick_indices)
    axes[2].set_xticklabels([test_ends[i] for i in tick_indices],
                             rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--features",   required=True, help="Features CSV path")
    parser.add_argument("--mode",       default="directional",
                        choices=["directional", "binary", "multiclass"],
                        help="Model mode (default: directional)")
    parser.add_argument("--fold-size",  type=int, default=20,
                        help="Sessions per test fold (default: 20 ~ 1 month)")
    parser.add_argument("--out-dir",    default=".",
                        help="Output directory for report and plot")
    args = parser.parse_args()

    label_col  = LABEL_COLS[args.mode]
    binary     = args.mode in ("directional", "binary")
    directional = args.mode == "directional"
    symbol     = Path(args.features).stem.split("_features")[0].upper()
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.features}...")
    df = load(args.features, label_col, directional)
    print(f"  {len(df)} sessions loaded "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")

    n_folds = (len(df) - MIN_TRAIN) // args.fold_size
    print(f"  Expected folds: ~{n_folds}  (fold_size={args.fold_size}, "
          f"min_train={MIN_TRAIN})")

    print(f"\nRunning walk-forward ({args.mode} mode)...")
    folds = run_walk_forward(df, label_col, args.fold_size, binary)
    print(f"  Completed {len(folds)} folds")

    report = format_report(folds, symbol, args.mode, args.fold_size)
    print("\n" + report)

    report_path = out_dir / f"{symbol.lower()}_walkforward_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    plot_path = str(out_dir / f"{symbol.lower()}_walkforward.png")
    plot_results(folds, symbol, args.mode, plot_path)


if __name__ == "__main__":
    main()
