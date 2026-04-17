"""
ml/models/breakout/feature_importance.py

Train XGBoost on full dataset and report feature importances.
For inspection only — not a validated model.

Usage:
    python -m ml.models.breakout.feature_importance \
        --features data/features/breakout/features.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from ml.models.breakout.config import ModelConfig
from ml.models.breakout.validate import FEATURES, LABEL_COL


def main():
    parser = argparse.ArgumentParser(description="Breakout feature importance analysis")
    parser.add_argument("--features", required=True)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config YAML")
    parser.add_argument("--timeframe", type=str, default=None,
                        help="Single timeframe (removes timeframe_encoded from features)")
    args = parser.parse_args()

    cfg = ModelConfig.from_yaml(args.config) if args.config else None
    timeframe = args.timeframe or (cfg.timeframe if cfg else None)

    features = list(FEATURES)
    if timeframe:
        features = [f for f in features if f != "timeframe_encoded"]

    df = pd.read_csv(args.features, parse_dates=["ts"])
    df = df.dropna(subset=[LABEL_COL])
    print(f"Loaded {len(df)} candidates")

    X = df[features]
    y = df[LABEL_COL]
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
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y, sample_weight=weights)

    # Extract importances by gain
    importance = model.get_booster().get_score(importance_type="gain")
    # Map f0, f1, ... back to feature names
    feat_map = {f"f{i}": name for i, name in enumerate(features)}
    ranked = sorted(
        [(feat_map.get(k, k), v) for k, v in importance.items()],
        key=lambda x: x[1], reverse=True,
    )

    # Text report
    lines = ["FEATURE IMPORTANCE (by gain)", "=" * 50, ""]
    lines.append(f"{'Rank':>4}  {'Feature':<35}  {'Gain':>10}")
    lines.append("-" * 55)
    for i, (name, gain) in enumerate(ranked, 1):
        lines.append(f"{i:>4}  {name:<35}  {gain:>10.1f}")
    report = "\n".join(lines)
    print("\n" + report)

    if timeframe:
        out_dir = Path(f"data/reports/breakout/{timeframe}")
    else:
        out_dir = Path("data/reports/breakout/combined")
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "feature_importance.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    # Plot top 20
    top = ranked[:20]
    names = [t[0] for t in reversed(top)]
    gains = [t[1] for t in reversed(top)]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(names, gains, color="#00bcd4")
    ax.set_xlabel("Gain")
    ax.set_title("Breakout Entry Quality — Top 20 Features by Gain")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    ax.tick_params(colors="#e0e0e0")
    ax.xaxis.label.set_color("#e0e0e0")
    ax.title.set_color("#e0e0e0")
    for spine in ax.spines.values():
        spine.set_color("#2a2a4a")

    plt.tight_layout()
    plot_path = out_dir / "feature_importance.png"
    plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
