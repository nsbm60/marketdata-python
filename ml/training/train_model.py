"""
train_model.py

Train an XGBoost classifier on the session feature matrix to predict
first-hour session type: trend / containment / reversal / double_sweep.

Usage:
    python train_model.py --features nvda_features.csv
    python train_model.py --features nvda_features.csv --test-sessions 110
    python train_model.py --features nvda_features.csv --binary
    python train_model.py --features nvda_features.csv --directional

Outputs:
    <symbol>_model.json      -- trained XGBoost model
    <symbol>_report.txt      -- evaluation report

Dependencies:
    pip install xgboost scikit-learn pandas numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

# Features available by 10:30am on the session date -- no lookahead
FEATURES = [
    # Gap / prior session context (known at 9:30)
    "gap_pct",
    "prior_day_range_pct",
    "atr20",

    # First-hour price structure (known at 10:30)
    "fh_range_pct",
    "fh_range_atr",
    "fh_vwap_dev",

    # First-15-minute structure (known at 9:45)
    "f15_range_ratio",
    "f15_vol_ratio",

    # Volume
    "fh_vol_ratio",

    # Sweep signal (known at 10:30)
    "sweep_signal",
    "sweep_direction",

    # Regime features (rolling, lagged -- no lookahead)
    "rolling_reversal_rate",
    "rolling_high_set_rate",
    "directional_bias",
    "gap_regime_alignment",
]

# Columns deliberately excluded and why:
#   fh_high_is_session_high / fh_low_is_session_low -- used to compute the label
#   fh_range_abs  -- redundant with fh_range_pct and fh_range_atr
#   avg_vol_20    -- raw volume number, already normalized into fh_vol_ratio
#   date          -- not a feature
#   label / label_name -- target

LABEL_COL           = "label"
BINARY_LABEL_COL    = "binary_label"
DIRECTIONAL_LABEL_COL = "directional_label"
LABEL_NAMES         = {0: "trend", 1: "containment", 2: "reversal", 3: "double_sweep"}
BINARY_NAMES        = {0: "non_reversal", 1: "reversal"}
DIRECTIONAL_NAMES   = {0: "buy_the_dip", 1: "fade_the_high"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_and_validate(path: str, mode: str) -> pd.DataFrame:
    """
    Load feature CSV and compute derived labels.

    mode: 'multiclass', 'binary', or 'directional'
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    if LABEL_COL not in df.columns:
        print(f"ERROR: Label column '{LABEL_COL}' not found.")
        sys.exit(1)

    # Compute derived labels
    # Binary: 1 = reversal or double_sweep, 0 = trend or containment
    df[BINARY_LABEL_COL] = df[LABEL_COL].isin([2, 3]).astype(int)

    # Directional: for reversal sessions (label=2), predict direction
    # 1 = fade_the_high (fh_high_is_session_high), 0 = buy_the_dip (fh_low_is_session_low)
    df[DIRECTIONAL_LABEL_COL] = df["fh_high_is_session_high"]

    # Determine which label column to use
    if mode == "directional":
        label_col = DIRECTIONAL_LABEL_COL
    elif mode == "binary":
        label_col = BINARY_LABEL_COL
    else:
        label_col = LABEL_COL

    before = len(df)
    df = df.dropna(subset=FEATURES + [label_col])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with nulls (rolling warmup)")

    if mode == "directional":
        # Keep only reversal sessions (label=2) for directional prediction
        before = len(df)
        df = df[df[LABEL_COL] == 2].copy()
        print(f"Directional mode: kept {len(df)} reversal sessions (dropped {before - len(df)} non-reversal)")

    return df, label_col


def chronological_split(df: pd.DataFrame, n_test: int):
    """Time-aware split -- test set is the most recent n_test sessions."""
    train = df.iloc[:-n_test].copy()
    test  = df.iloc[-n_test:].copy()
    return train, test


def class_weight_summary(y: pd.Series, name_map: dict) -> dict:
    counts = y.value_counts().sort_index()
    total  = len(y)
    return {name_map[int(k)]: f"{v} ({100*v/total:.1f}%)" for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(df_train: pd.DataFrame, label_col: str, binary: bool) -> xgb.XGBClassifier:
    X = df_train[FEATURES]
    y = df_train[label_col]

    # Address class imbalance with sample weights
    # This gives the model equal effective exposure to each class
    weights = compute_sample_weight(class_weight="balanced", y=y)

    objective   = "binary:logistic" if binary else "multi:softprob"
    eval_metric = "logloss"         if binary else "mlogloss"
    extra_args  = {}                if binary else {"num_class": 4}

    model = xgb.XGBClassifier(
        objective        = objective,
        eval_metric      = eval_metric,
        **extra_args,
        n_estimators     = 300,
        max_depth        = 4,        # shallow -- we have ~400 training rows
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,        # prevents splits on very small classes

        early_stopping_rounds = 30,
        random_state     = 42,
        verbosity        = 0,
    )

    # Use 15% of training data as internal validation for early stopping
    n_val  = max(20, int(len(df_train) * 0.15))
    X_tr   = X.iloc[:-n_val]
    y_tr   = y.iloc[:-n_val]
    w_tr   = weights[:-n_val]
    X_val  = X.iloc[-n_val:]
    y_val  = y.iloc[-n_val:]

    model.fit(
        X_tr, y_tr,
        sample_weight    = w_tr,
        eval_set         = [(X_val, y_val)],
        verbose          = False,
    )

    print(f"Best iteration: {model.best_iteration}  "
          f"(val logloss: {model.best_score:.4f})")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: xgb.XGBClassifier,
             df_test: pd.DataFrame,
             df_train: pd.DataFrame,
             label_col: str,
             binary: bool,
             directional: bool = False) -> str:

    X_test = df_test[FEATURES]
    y_test = df_test[label_col]
    y_pred = model.predict(X_test)

    if directional:
        label_list = [0, 1]
        label_strs = [DIRECTIONAL_NAMES[i] for i in label_list]
    elif binary:
        label_list = [0, 1]
        label_strs = [BINARY_NAMES[i] for i in label_list]
    else:
        label_list = [0, 1, 2, 3]
        label_strs = [LABEL_NAMES[i] for i in label_list]

    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Train sessions: {len(df_train)}  "
                 f"({df_train['date'].min().date()} → {df_train['date'].max().date()})")
    lines.append(f"Test  sessions: {len(df_test)}  "
                 f"({df_test['date'].min().date()} → {df_test['date'].max().date()})")

    name_map = DIRECTIONAL_NAMES if directional else (BINARY_NAMES if binary else LABEL_NAMES)
    lines.append("\nTrain label distribution:")
    for name, summary in class_weight_summary(df_train[label_col], name_map).items():
        lines.append(f"  {name:<15} {summary}")

    lines.append("\nTest label distribution:")
    for name, summary in class_weight_summary(df_test[label_col], name_map).items():
        lines.append(f"  {name:<15} {summary}")

    lines.append("\nClassification report:")
    lines.append(classification_report(
        y_test, y_pred,
        labels      = label_list,
        target_names= label_strs,
        zero_division= 0,
    ))

    lines.append("Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred, labels=label_list)
    header = f"{'':>15}" + "".join(f"{n:>14}" for n in label_strs)
    lines.append(header)
    for i, row in enumerate(cm):
        lines.append(f"{label_strs[i]:>15}" + "".join(f"{v:>14}" for v in row))

    lines.append("\nFeature importance (top 10 by gain):")
    importance = model.get_booster().get_score(importance_type="gain")
    sorted_imp  = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_imp[:10]:
        lines.append(f"  {feat:<30} {score:.2f}")

    # Baseline: always predict majority class
    majority_class = df_train[label_col].mode()[0]
    baseline_acc   = (y_test == majority_class).mean()
    model_acc      = (y_pred == y_test.values).mean()
    lines.append(f"\nModel accuracy:    {model_acc:.3f}")
    lines.append(f"Baseline accuracy: {baseline_acc:.3f}  (always predict '{name_map[int(majority_class)]}')")
    lines.append(f"Lift over baseline: {model_acc - baseline_acc:+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost session classifier")
    parser.add_argument("--features",      required=True, help="Path to features CSV")
    parser.add_argument("--test-sessions", type=int, default=110,
                        help="Number of most-recent sessions to hold out for testing (default: 110)")
    parser.add_argument("--out-dir",       default=".", help="Output directory for model and report")
    parser.add_argument("--binary",        action="store_true", help="Use binary label (reversal vs non-reversal)")
    parser.add_argument("--directional",   action="store_true", help="Predict reversal direction: fade-the-high vs buy-the-dip (reversal sessions only)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Infer symbol from filename
    symbol = Path(args.features).stem.replace("_features", "").upper()

    # Determine mode
    if args.directional:
        mode = "directional"
        mode_str = "DIRECTIONAL"
    elif args.binary:
        mode = "binary"
        mode_str = "BINARY"
    else:
        mode = "multiclass"
        mode_str = "4-CLASS"

    print(f"Loading features from {args.features}")
    df, label_col = load_and_validate(args.features, mode)
    print(f"  {len(df)} sessions loaded ({df['date'].min().date()} → {df['date'].max().date()})")
    print(f"Mode: {mode_str}  (label column: {label_col})")

    binary      = args.binary or args.directional  # both use binary classification
    directional = args.directional

    # Adjust test sessions for directional mode (fewer total sessions)
    test_sessions = args.test_sessions
    if directional and test_sessions > len(df) * 0.3:
        test_sessions = int(len(df) * 0.25)
        print(f"Adjusted test sessions to {test_sessions} (25% of reversal sessions)")

    print(f"\nSplitting: {len(df) - test_sessions} train / {test_sessions} test")
    df_train, df_test = chronological_split(df, test_sessions)

    print("\nTraining...")
    model = train(df_train, label_col, binary)

    print("\nEvaluating...")
    report = evaluate(model, df_test, df_train, label_col, binary, directional)
    print("\n" + report)

    # Save model
    suffix     = "directional" if directional else ("binary" if args.binary else "multiclass")
    model_path = out_dir / f"{symbol.lower()}_{suffix}_model.json"
    model.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Save report
    report_path = out_dir / f"{symbol.lower()}_{suffix}_report.txt"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
