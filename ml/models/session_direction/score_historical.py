"""
score_historical.py

Runs the stage 1 session direction model retrospectively across all
historical sessions. Output is used as stage 2 input features.

Usage:
    python ml/models/session_direction/score_historical.py --symbols NVDA AMD
    python ml/models/session_direction/score_historical.py --symbols NVDA AMD --window 60
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from ml.shared.paths import features_path, model_path
from ml.shared.features import FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def score_historical(symbol: str, window: int = 60) -> None:
    """
    Score all historical sessions for a symbol and save output.
    """
    # Load feature matrix
    feat_path = features_path(symbol, window)
    if not feat_path.exists():
        log.error(f"Feature matrix not found: {feat_path}")
        return

    df = pd.read_csv(feat_path, parse_dates=["date"])
    log.info(f"{symbol}: loaded {len(df)} sessions from {feat_path}")

    # Verify required columns exist
    required_cols = ["close_position", "w_high", "w_low", "open_930"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        log.error(f"{symbol}: missing required columns: {missing}")
        log.error("Regenerate feature matrix with: python ml/etl/build_feature_matrix.py --symbol {symbol}")
        return

    # Load model
    m_path = model_path(symbol, "session_direction", window)
    if not m_path.exists():
        log.error(f"Model not found: {m_path}")
        return

    model = xgb.XGBClassifier()
    model.load_model(str(m_path))
    log.info(f"{symbol}: loaded model from {m_path}")

    # Score all sessions
    X = df[FEATURES]
    probs = model.predict_proba(X)
    labels = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    # Compute entry_quality
    # label=1 (fade_the_high): close_position = how much opportunity remains
    # label=0 (buy_the_dip): 1 - close_position = how much opportunity remains
    entry_quality = [
        float(df["close_position"].iloc[i]) if labels[i] == 1
        else float(1.0 - df["close_position"].iloc[i])
        for i in range(len(df))
    ]

    # Build output DataFrame
    output = pd.DataFrame({
        "session_date":     df["date"].dt.date,
        "label":            labels.astype(int),
        "confidence":       confidences.round(4),
        "first_hour_high":  df["w_high"].round(4),
        "first_hour_low":   df["w_low"].round(4),
        "first_hour_range": (df["w_high"] - df["w_low"]).round(4),
        "open_930":         df["open_930"].round(4),
        "entry_quality":    [round(eq, 4) for eq in entry_quality],
    })

    # Save output
    out_path = features_path(symbol, window, prefix="session_direction")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    log.info(f"{symbol}: wrote {len(output)} rows to {out_path}")

    # Summary
    n_fade = (output["label"] == 1).sum()
    n_dip = (output["label"] == 0).sum()
    avg_conf = output["confidence"].mean()
    avg_eq = output["entry_quality"].mean()
    log.info(f"{symbol}: fade_the_high={n_fade} buy_the_dip={n_dip} "
             f"avg_confidence={avg_conf:.3f} avg_entry_quality={avg_eq:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Retrospective stage 1 scoring"
    )
    parser.add_argument("--symbols", nargs="+", required=True,
                        help="Symbols to score")
    parser.add_argument("--window", type=int, default=60,
                        help="Prediction window minutes (default: 60)")
    args = parser.parse_args()

    for symbol in [s.upper() for s in args.symbols]:
        log.info(f"Scoring {symbol}...")
        score_historical(symbol, args.window)

    log.info("Done.")


if __name__ == "__main__":
    main()
