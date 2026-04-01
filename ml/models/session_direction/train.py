"""
train.py

Train an XGBoost classifier on the session feature matrix to predict
first-hour session type: trend / containment / reversal / double_sweep.

Usage:
    python -m ml.models.session_direction.train --features data/features/NVDA/features_w60.csv
    python -m ml.models.session_direction.train --features data/features/NVDA/features_w60.csv --test-sessions 110

Outputs:
    data/models/{SYMBOL}/session_direction_w{window}.json  -- trained XGBoost model
    data/models/{SYMBOL}/calibrated_w{window}.pkl          -- calibration model
    data/reports/{SYMBOL}/directional_w{window}_report.txt -- evaluation report

Dependencies:
    pip install xgboost scikit-learn pandas numpy
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight

# Import shared modules
from ml.shared.features import FEATURES
from ml.shared.constants import (
    LABEL_COL, BINARY_LABEL_COL, DIRECTIONAL_LABEL_COL,
    LABEL_NAMES, BINARY_NAMES, DIRECTIONAL_NAMES,
)
from ml.shared.paths import model_path, calibration_path, report_path, ensure_dirs


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_and_validate(path: str, label_col: str, directional: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    if label_col not in df.columns:
        print(f"ERROR: Label column '{label_col}' not found. Re-run build_feature_matrix.py to regenerate the CSV.")
        sys.exit(1)

    # Fill nulls in correlation features with column median before dropping.
    # These arise from rolling window warmup at the start of the dataset.
    corr_features = ["target_qqq_corr", "target_smh_corr", "target_qqq_beta"]
    for col in corr_features:
        if col in df.columns and df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)

    before = len(df)
    df = df.dropna(subset=FEATURES + [label_col])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with nulls (rolling warmup or non-directional sessions)")

    if directional:
        before = len(df)
        df = df[df[label_col].isin([0, 1])].copy()
        print(f"Directional mode: kept {len(df)} reversal sessions (dropped {before - len(df)} double_sweep/trend/containment)")

    return df


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
          f"(val mlogloss: {model.best_score:.4f})")

    # Return validation set for calibration -- not used for tree fitting
    return model, X_val, y_val


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(model, X_cal, y_cal):
    """
    Fit an isotonic regression calibrator on top of the trained XGBoost model.
    Uses the early-stopping validation set, which was not used for tree fitting.

    Maps raw XGBoost class-1 probabilities to empirical probabilities so that
    a predicted confidence of 0.75 actually corresponds to ~75% accuracy.
    Returns the fitted IsotonicRegression object.
    """
    raw_probs = model.predict_proba(X_cal)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs[:, 1], y_cal.values.astype(int))
    return iso


def apply_calibration(model, iso: IsotonicRegression, X: pd.DataFrame):
    """Apply calibration to get adjusted probabilities. Returns (class0_prob, class1_prob)."""
    raw_probs = model.predict_proba(X)
    cal_p1    = iso.predict(raw_probs[:, 1])
    cal_p0    = 1 - cal_p1
    return np.column_stack([cal_p0, cal_p1])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model,
             df_test: pd.DataFrame,
             df_train: pd.DataFrame,
             label_col: str,
             binary: bool,
             directional: bool = False,
             calibrated_model=None) -> str:

    X_test      = df_test[FEATURES]
    y_test      = df_test[label_col]
    # For predictions, use calibrated probabilities if available
    if calibrated_model is not None:
        cal_probs_pred = apply_calibration(model, calibrated_model, X_test)
        y_pred = cal_probs_pred.argmax(axis=1)
    else:
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

    # Calibration quality section
    if calibrated_model is not None:
        raw_probs = model.predict_proba(X_test)
        cal_probs = apply_calibration(model, calibrated_model, X_test)
        y_arr     = y_test.values.astype(int)

        raw_brier = brier_score_loss(y_arr, raw_probs[:, 1])
        cal_brier = brier_score_loss(y_arr, cal_probs[:, 1])
        lines.append(f"\nCalibration (Brier score -- lower is better):")
        lines.append(f"  Raw model:   {raw_brier:.4f}")
        lines.append(f"  Calibrated:  {cal_brier:.4f}  ({(raw_brier-cal_brier)/raw_brier*100:+.1f}%)")

        confidence = cal_probs.max(axis=1)
        predicted  = cal_probs.argmax(axis=1)
        correct    = (predicted == y_arr)
        lines.append(f"\nCalibrated confidence threshold analysis:")
        lines.append(f"  {'Threshold':>10}  {'Sessions':>9}  {'Coverage':>9}  {'Accuracy':>9}")
        lines.append("  " + "-" * 43)
        for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            mask = confidence >= thr
            n    = mask.sum()
            cov  = n / len(y_arr)
            acc  = correct[mask].mean() if n > 0 else float("nan")
            lines.append(f"  {thr:>10.2f}  {n:>9}  {cov:>8.1%}  {acc:>8.1%}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import pickle
    import re

    parser = argparse.ArgumentParser(description="Train XGBoost session classifier")
    parser.add_argument("--features",      required=True, help="Path to features CSV")
    parser.add_argument("--symbol",        help="Symbol (if not inferrable from path)")
    parser.add_argument("--test-sessions", type=int, default=110,
                        help="Number of most-recent sessions to hold out for testing (default: 110)")
    parser.add_argument("--binary",        action="store_true", help="Use binary label (reversal vs non-reversal)")
    parser.add_argument("--directional",   action="store_true", help="Predict reversal direction: fade-the-high vs buy-the-dip (reversal sessions only)")
    args = parser.parse_args()

    # Infer symbol and window from filename or path
    # Handles: data/features/NVDA/features_w60.csv or nvda_features_w60.csv
    features_path = Path(args.features)
    filename = features_path.stem

    # Try to extract window from filename (e.g., features_w60 -> 60)
    match = re.search(r"_w(\d+)$", filename)
    window = int(match.group(1)) if match else 60

    # Extract symbol: explicit arg > parent dir > filename
    if args.symbol:
        symbol = args.symbol.upper()
    elif features_path.parent.name.isupper() and features_path.parent.name.isalpha():
        # Parent directory looks like a symbol (e.g., NVDA, AMD, GOOGL)
        symbol = features_path.parent.name
    else:
        # Try to extract from filename: nvda_features_w60 -> NVDA
        symbol = filename.replace(f"_w{window}", "").replace("features", "").replace("_", "").upper()
        if not symbol:
            print("ERROR: Could not determine symbol. Use --symbol argument.")
            sys.exit(1)

    print(f"Loading features from {args.features}")
    print(f"Symbol: {symbol}, Window: {window}")
    _label = DIRECTIONAL_LABEL_COL if args.directional else (BINARY_LABEL_COL if args.binary else LABEL_COL)
    df = load_and_validate(args.features, _label, directional=args.directional)
    print(f"  {len(df)} sessions loaded ({df['date'].min().date()} → {df['date'].max().date()})")

    binary      = args.binary
    directional = args.directional

    if directional:
        label_col = DIRECTIONAL_LABEL_COL
        mode_str  = "DIRECTIONAL"
    elif binary:
        label_col = BINARY_LABEL_COL
        mode_str  = "BINARY"
    else:
        label_col = LABEL_COL
        mode_str  = "4-CLASS"
    print(f"Mode: {mode_str}  (label column: {label_col})")

    print(f"\nSplitting: {len(df) - args.test_sessions} train / {args.test_sessions} test")
    df_train, df_test = chronological_split(df, args.test_sessions)

    print("\nTraining...")
    model, X_cal, y_cal = train(df_train, label_col, binary or directional)

    print("\nCalibrating probabilities...")
    calibrated = calibrate(model, X_cal, y_cal)

    print("\nEvaluating...")
    report = evaluate(model, df_test, df_train, label_col, binary, directional,
                      calibrated_model=calibrated)
    print("\n" + report)

    # Ensure output directories exist
    ensure_dirs(symbol)

    # Save model using new path structure
    model_file = model_path(symbol, "session_direction", window)
    model.save_model(str(model_file))
    print(f"\nXGBoost model saved to {model_file}")

    # Save calibration model
    cal_file = calibration_path(symbol, window)
    with open(str(cal_file), "wb") as fh:
        pickle.dump(calibrated, fh)
    print(f"Calibrated model saved to {cal_file}")

    # Save report
    report_file = report_path(symbol, f"directional_w{window}")
    report_file.write_text(report)
    print(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()
