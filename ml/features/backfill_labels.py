"""
backfill_labels.py

Backfill session_labels table from existing feature CSVs.

Reads feature CSVs from data/ directory and populates the ClickHouse
session_labels table. Idempotent -- safe to run multiple times since
ReplacingMergeTree handles deduplication on (symbol, session_date, window_minutes).

Usage:
    # Backfill from a specific CSV
    python ml/features/backfill_labels.py --csv data/nvda_features_w60.csv --symbol NVDA

    # Backfill all feature CSVs in data/ directory
    python ml/features/backfill_labels.py --all

Dependencies:
    pip install clickhouse-connect pandas python-dotenv pyzmq
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import clickhouse_connect
import pandas as pd
from dotenv import load_dotenv

from discovery import ServiceLocator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BATCH_SIZE = 1000

# Columns to extract from feature CSV
CSV_COLUMNS = [
    "date",
    "label",
    "label_name",
    "binary_label",
    "directional_label",
    "fh_high_is_session_high",
    "fh_low_is_session_low",
]


def get_ch_client():
    """Get ClickHouse client via service discovery."""
    log.info("Discovering ClickHouse...")
    ch_endpoint = ServiceLocator.wait_for_service(
        ServiceLocator.CLICKHOUSE,
        timeout_sec=60,
    )
    return clickhouse_connect.get_client(
        host     = ch_endpoint.host,
        port     = ch_endpoint.port,
        username = os.environ.get("CLICKHOUSE_USER",     "default"),
        password = os.environ.get("CLICKHOUSE_PASSWORD", "Aector99"),
        database = os.environ.get("CLICKHOUSE_DATABASE", "trading"),
    )


def infer_symbol_and_window(csv_path: str) -> tuple[str, int]:
    """
    Infer symbol and window from CSV filename.
    Expected patterns:
        nvda_features_w60.csv  -> (NVDA, 60)
        amd_features_w30.csv   -> (AMD, 30)
        nvda_features.csv      -> (NVDA, 60)  # default window
    """
    name = Path(csv_path).stem

    # Try to match pattern like "nvda_features_w60"
    match = re.match(r"([a-zA-Z]+)_features(?:_w(\d+))?", name)
    if match:
        symbol = match.group(1).upper()
        window = int(match.group(2)) if match.group(2) else 60
        return symbol, window

    # Fallback: just use the first part as symbol
    parts = name.split("_")
    return parts[0].upper(), 60


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load feature CSV and extract label columns."""
    df = pd.read_csv(csv_path, parse_dates=["date"])

    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[CSV_COLUMNS].copy()


def backfill_from_csv(ch_client, csv_path: str, symbol: str = None):
    """
    Backfill session_labels from a single CSV file.
    """
    # Infer symbol and window if not provided
    inferred_symbol, window = infer_symbol_and_window(csv_path)
    symbol = symbol or inferred_symbol

    log.info(f"Loading {csv_path} for {symbol} (window={window})")

    df = load_csv(csv_path)
    log.info(f"  {len(df)} rows loaded")

    # Prepare data for insertion
    rows = []
    for _, row in df.iterrows():
        # Handle NaN in directional_label -> None
        directional = row["directional_label"]
        if pd.isna(directional):
            directional = None
        else:
            directional = int(directional)

        rows.append([
            symbol,
            row["date"].date(),
            int(row["label"]),
            row["label_name"],
            int(row["binary_label"]),
            directional,
            int(row["fh_high_is_session_high"]),
            int(row["fh_low_is_session_low"]),
            window,
        ])

    # Insert in batches
    columns = [
        "symbol",
        "session_date",
        "label",
        "label_name",
        "binary_label",
        "directional_label",
        "fh_high_is_session_high",
        "fh_low_is_session_low",
        "window_minutes",
    ]

    total_inserted = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        ch_client.insert(
            table="session_labels",
            data=batch,
            column_names=columns,
        )
        total_inserted += len(batch)
        if total_inserted % 5000 == 0 or total_inserted == len(rows):
            log.info(f"  {total_inserted:,} rows inserted...")

    log.info(f"  Backfill complete: {total_inserted:,} rows for {symbol} (w{window})")
    return total_inserted


def find_feature_csvs(data_dir: str = "data") -> list[str]:
    """Find all feature CSV files in the data directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    csvs = []
    for f in data_path.glob("*_features*.csv"):
        csvs.append(str(f))

    return sorted(csvs)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill session_labels from feature CSVs"
    )
    parser.add_argument("--csv", help="Path to a specific feature CSV")
    parser.add_argument("--symbol", help="Symbol (inferred from filename if not provided)")
    parser.add_argument("--all", action="store_true",
                        help="Backfill all feature CSVs in data/ directory")
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing feature CSVs (default: data/)")
    args = parser.parse_args()

    if not args.csv and not args.all:
        parser.error("Either --csv or --all is required")

    ch_client = get_ch_client()

    total = 0

    if args.all:
        csvs = find_feature_csvs(args.data_dir)
        if not csvs:
            log.warning(f"No feature CSVs found in {args.data_dir}")
            return

        log.info(f"Found {len(csvs)} feature CSVs to backfill")
        for csv_path in csvs:
            try:
                count = backfill_from_csv(ch_client, csv_path)
                total += count
            except Exception as e:
                log.error(f"Failed to backfill {csv_path}: {e}")
    else:
        total = backfill_from_csv(ch_client, args.csv, args.symbol)

    log.info(f"Total rows inserted: {total:,}")


if __name__ == "__main__":
    main()
