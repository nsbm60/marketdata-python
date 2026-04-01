"""
run_pipeline.py

Orchestrates the full ML pipeline for one or more symbols:
0. ETL -- pull historical bars from Alpaca (skip with --skip-etl)
1. Build feature matrices (all windows: 15, 30, 45, 60)
2. Train session direction model (directional mode, w60)
3. Score historical sessions (stage 2 input)
4. Backfill session labels (if needed)
5. Run confidence exploration

Usage:
    python ml/pipeline/run_pipeline.py --symbols NVDA AMD --skip-etl
    python ml/pipeline/run_pipeline.py --symbols MSFT AAPL GOOGL
    python ml/pipeline/run_pipeline.py --symbols TSLA --windows 60
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.shared.paths import features_path, model_path, ensure_dirs
from ml.shared.clickhouse import get_ch_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Base directory for running scripts
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Lazy-initialized ClickHouse client
_ch_client = None


def get_pipeline_ch_client():
    """Get ClickHouse client for pipeline checks."""
    global _ch_client
    if _ch_client is None:
        _ch_client = get_ch_client()
    return _ch_client


def needs_label_backfill(symbol: str) -> bool:
    """Check if session_labels need to be backfilled for a symbol."""
    ch = get_pipeline_ch_client()
    result = ch.query(
        """
        SELECT count() FROM session_labels
        WHERE symbol = %(symbol)s AND window_minutes = 60
        """,
        parameters={"symbol": symbol}
    )
    count = result.result_rows[0][0] if result.result_rows else 0
    return count < 100  # Need at least 100 sessions for meaningful analysis


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    log.info(f"Running: {description}")
    log.info(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=False,  # Stream output to console
            text=True,
        )
        if result.returncode != 0:
            log.error(f"  FAILED with exit code {result.returncode}")
            return False
        log.info(f"  SUCCESS")
        return True
    except Exception as e:
        log.error(f"  ERROR: {e}")
        return False


def run_pipeline_for_symbol(
    symbol: str,
    windows: list[int],
    skip_etl: bool = False,
    skip_features: bool = False,
    skip_train: bool = False,
    skip_score: bool = False,
    skip_explore: bool = False,
) -> dict:
    """
    Run the full pipeline for a single symbol.

    Returns dict with status of each step.
    """
    symbol = symbol.upper()
    log.info(f"\n{'='*60}")
    log.info(f"PIPELINE: {symbol}")
    log.info(f"{'='*60}\n")

    ensure_dirs(symbol)
    results = {"symbol": symbol}

    # Step 0: ETL -- pull historical bars from Alpaca
    if not skip_etl:
        cmd = [
            sys.executable,
            "ml/etl/alpaca_bars_etl.py",
            "--symbol", symbol,
            "--start", "2021-01-01",
        ]
        if run_command(cmd, f"ETL historical bars for {symbol}"):
            results["etl"] = "OK"
        else:
            results["etl"] = "FAILED"
            return results
    else:
        results["etl"] = "SKIPPED"

    # Step 1: Build feature matrices
    if not skip_features:
        if len(windows) == 4:
            cmd = [
                sys.executable,
                "ml/etl/build_feature_matrix.py",
                "--symbol", symbol,
                "--all-windows",
            ]
        else:
            # Build each window separately
            for w in windows:
                cmd = [
                    sys.executable,
                    "ml/etl/build_feature_matrix.py",
                    "--symbol", symbol,
                    "--window", str(w),
                ]
                if not run_command(cmd, f"Build features w{w} for {symbol}"):
                    results["features"] = "FAILED"
                    return results
            results["features"] = "OK"

        if len(windows) == 4:
            if run_command(cmd, f"Build all feature matrices for {symbol}"):
                results["features"] = "OK"
            else:
                results["features"] = "FAILED"
                return results
    else:
        results["features"] = "SKIPPED"

    # Step 2: Train session direction model (w60, directional)
    if not skip_train:
        features_file = features_path(symbol, 60)
        if not features_file.exists():
            log.error(f"Feature file not found: {features_file}")
            results["train"] = "FAILED"
            return results

        cmd = [
            sys.executable,
            "ml/models/session_direction/train.py",
            "--features", str(features_file),
            "--directional",
        ]
        if run_command(cmd, f"Train session direction model for {symbol}"):
            results["train"] = "OK"
        else:
            results["train"] = "FAILED"
            return results
    else:
        results["train"] = "SKIPPED"

    # Step 3: Score historical sessions
    if not skip_score:
        cmd = [
            sys.executable,
            "ml/models/session_direction/score_historical.py",
            "--symbols", symbol,
        ]
        if run_command(cmd, f"Score historical sessions for {symbol}"):
            results["score"] = "OK"
        else:
            results["score"] = "FAILED"
            return results
    else:
        results["score"] = "SKIPPED"

    # Step 4: Backfill session labels if needed
    if needs_label_backfill(symbol):
        features_file = features_path(symbol, 60)
        cmd = [
            sys.executable,
            "ml/etl/backfill_labels.py",
            "--csv", str(features_file),
            "--symbol", symbol,
        ]
        if run_command(cmd, f"Backfill session labels for {symbol}"):
            results["labels"] = "OK"
        else:
            results["labels"] = "FAILED"
            return results
    else:
        log.info(f"{symbol}: session labels already exist")
        results["labels"] = "SKIPPED"

    # Step 5: Run confidence exploration (optional)
    if not skip_explore and len(windows) > 1:
        cmd = [
            sys.executable,
            "ml/models/session_direction/explore_confidence.py",
            "--symbols", symbol,
        ]
        if run_command(cmd, f"Explore confidence for {symbol}"):
            results["explore"] = "OK"
        else:
            results["explore"] = "FAILED"
    else:
        results["explore"] = "SKIPPED"

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run full ML pipeline for symbols"
    )
    parser.add_argument("--symbols", nargs="+", required=True,
                        help="Symbols to process")
    parser.add_argument("--windows", nargs="+", type=int, default=[15, 30, 45, 60],
                        help="Windows to build (default: 15 30 45 60)")
    parser.add_argument("--skip-etl", action="store_true",
                        help="Skip Alpaca bar backfill (use if bars already in ClickHouse)")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature matrix generation")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model training")
    parser.add_argument("--skip-score", action="store_true",
                        help="Skip historical scoring")
    parser.add_argument("--skip-explore", action="store_true",
                        help="Skip confidence exploration")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols]
    windows = sorted(args.windows)

    log.info(f"Pipeline starting for {len(symbols)} symbols: {symbols}")
    log.info(f"Windows: {windows}")
    log.info(f"Skip flags: etl={args.skip_etl}, features={args.skip_features}, "
             f"train={args.skip_train}, score={args.skip_score}, explore={args.skip_explore}")

    all_results = []
    for symbol in symbols:
        results = run_pipeline_for_symbol(
            symbol=symbol,
            windows=windows,
            skip_etl=args.skip_etl,
            skip_features=args.skip_features,
            skip_train=args.skip_train,
            skip_score=args.skip_score,
            skip_explore=args.skip_explore,
        )
        all_results.append(results)

    # Print summary
    print(f"\n{'='*70}")
    print("PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Symbol':<8} {'ETL':<8} {'Features':<10} {'Train':<8} {'Score':<8} {'Labels':<8} {'Explore':<8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['symbol']:<8} {r.get('etl', '-'):<8} {r.get('features', '-'):<10} "
              f"{r.get('train', '-'):<8} {r.get('score', '-'):<8} "
              f"{r.get('labels', '-'):<8} {r.get('explore', '-'):<8}")

    # Check for critical failures (ETL, Features, Train, Score, Labels)
    # Explore failures are not critical
    critical_keys = ["etl", "features", "train", "score", "labels"]
    failures = [r["symbol"] for r in all_results
                if any(r.get(k) == "FAILED" for k in critical_keys)]
    if failures:
        print(f"\nFAILED symbols: {failures}")
        sys.exit(1)
    else:
        print(f"\nAll {len(symbols)} symbols completed successfully.")


if __name__ == "__main__":
    main()
