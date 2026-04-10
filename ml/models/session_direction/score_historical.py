"""
score_historical.py

Runs the stage 1 session direction model retrospectively across all
historical sessions. Output is used as stage 2 input features.

Usage:
    # Score all historical sessions
    python ml/models/session_direction/score_historical.py --symbols NVDA AMD
    python ml/models/session_direction/score_historical.py --symbols NVDA AMD --window 60

    # Score a single session (prints what live scorer would publish)
    python ml/models/session_direction/score_historical.py --symbols NVDA --date today
    python ml/models/session_direction/score_historical.py --symbols NVDA --date 2026-03-31
"""

import argparse
import logging
import sys
from datetime import date, datetime
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


DIRECTION_NAMES = {0: "buy_the_dip", 1: "fade_the_high"}


def score_single_session(symbol: str, session_date: date, window: int = 60) -> None:
    """
    Score a single session and print what the live scorer would publish.
    """
    # Load feature matrix
    feat_path = features_path(symbol, window)
    if not feat_path.exists():
        log.error(f"Feature matrix not found: {feat_path}")
        return

    df = pd.read_csv(feat_path, parse_dates=["date"])

    # Filter to specific date
    df["session_date"] = df["date"].dt.date
    row = df[df["session_date"] == session_date]

    if row.empty:
        log.error(f"{symbol}: no data for {session_date}")
        log.info(f"Available dates: {df['session_date'].min()} to {df['session_date'].max()}")
        return

    row = row.iloc[0]

    # Load model
    m_path = model_path(symbol, "session_direction", window)
    if not m_path.exists():
        log.error(f"Model not found: {m_path}")
        return

    model = xgb.XGBClassifier()
    model.load_model(str(m_path))

    # Score
    X = pd.DataFrame([row[FEATURES]])
    probs = model.predict_proba(X)[0]
    label = int(probs.argmax())
    confidence = float(probs.max())
    direction = DIRECTION_NAMES[label]

    # Compute entry quality
    close_pos = float(row["close_position"])
    entry_quality = close_pos if label == 1 else (1.0 - close_pos)

    # First-hour levels
    w_high = float(row["w_high"])
    w_low = float(row["w_low"])
    w_range = w_high - w_low
    open_930 = float(row["open_930"])

    # Print formatted output
    print(f"\n{'='*60}")
    print(f"SESSION PREDICTION: {symbol} {session_date}")
    print(f"{'='*60}")
    print(f"  Window:          w{window}")
    print(f"  Prediction:      {direction}")
    print(f"  Label:           {label}")
    print(f"  Confidence:      {confidence:.1%}")
    print(f"")
    print(f"  First-Hour Levels:")
    print(f"    Open (9:30):   ${open_930:.2f}")
    print(f"    High:          ${w_high:.2f}")
    print(f"    Low:           ${w_low:.2f}")
    print(f"    Range:         ${w_range:.2f}")
    print(f"")
    print(f"  Entry Quality:   {entry_quality:.1%}")
    print(f"  Close Position:  {close_pos:.1%}")
    print(f"{'='*60}\n")


def parse_date(date_str: str) -> date:
    """Parse date string, supporting 'today' and YYYY-MM-DD."""
    if date_str.lower() == "today":
        return date.today()
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date: {date_str}. Use 'today' or YYYY-MM-DD.")


def main():
    parser = argparse.ArgumentParser(
        description="Retrospective stage 1 scoring"
    )
    parser.add_argument("--symbols", nargs="+", required=True,
                        help="Symbols to score")
    parser.add_argument("--window", type=int, default=60,
                        help="Prediction window minutes (default: 60)")
    parser.add_argument("--date", type=parse_date, default=None,
                        help="Score single session (use 'today' or YYYY-MM-DD)")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols]

    if args.date:
        # Single session mode
        for symbol in symbols:
            score_single_session(symbol, args.date, args.window)
    else:
        # Batch mode
        for symbol in symbols:
            log.info(f"Scoring {symbol}...")
            score_historical(symbol, args.window)
        log.info("Done.")


if __name__ == "__main__":
    main()
