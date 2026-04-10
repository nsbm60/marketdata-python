"""
label_session.py

Session labeling logic for ML training. Labels sessions based on how the
first-hour (or prediction window) high/low relate to the full session high/low.

Labels:
    0 = trend         : price breaks out beyond first-hour range, keeps going
    1 = containment   : price stays mostly within first-hour range
    2 = reversal      : one extreme of first-hour is also session extreme
    3 = double_sweep  : both first-hour extremes are session extremes

Directional labels (for reversal sessions only):
    1 = fade_the_high : first-hour high is session high (stock ran up early then faded)
    0 = buy_the_dip   : first-hour low is session low (stock dropped early then recovered)

Usage:
    python label_session.py --symbol NVDA
    python label_session.py --symbol NVDA --window 30
    python label_session.py --symbol NVDA --start 2024-01-01 --output labels.csv

    # Write labels to ClickHouse session_labels table
    python label_session.py --symbol NVDA --write-db
    python label_session.py --symbol NVDA --start 2026-03-01 --write-db

Dependencies:
    pip install clickhouse-connect pandas numpy python-dotenv pyzmq
"""

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import clickhouse_connect
import numpy as np
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

VALID_WINDOWS = [15, 30, 45, 60]

LABEL_NAMES = {
    0: "trend",
    1: "containment",
    2: "reversal",
    3: "double_sweep",
}

DIRECTIONAL_NAMES = {
    0: "buy_the_dip",    # first-hour LOW is session low -- stock dropped early, recovered
    1: "fade_the_high",  # first-hour HIGH is session high -- stock ran up early, faded
}


SESSION_SQL = """
WITH

regular AS (
    SELECT
        toDate(ts)              AS session_date,
        ts,
        open, high, low, close,
        volume,
        toHour(ts)              AS hr,
        toMinute(ts)            AS mn
    FROM stock_bar FINAL
    WHERE symbol    = %(symbol)s
      AND period    = '1m'
      AND session   = 1
      AND toDate(toTimezone(ts, 'America/New_York')) BETWEEN %(start)s AND %(end)s
),

full_session AS (
    SELECT
        session_date,
        argMin(open, ts)    AS open_930,
        argMax(close, ts)   AS close_400,
        max(high)           AS session_high,
        min(low)            AS session_low
    FROM regular
    GROUP BY session_date
),

window_agg AS (
    SELECT
        session_date,
        max(high)           AS w_high,
        min(low)            AS w_low
    FROM regular
    WHERE {window_filter}
    GROUP BY session_date
)

SELECT
    fs.session_date     AS session_date,
    fs.open_930,
    fs.close_400,
    fs.session_high,
    fs.session_low,
    w.w_high,
    w.w_low
FROM full_session fs
JOIN window_agg w ON fs.session_date = w.session_date
ORDER BY fs.session_date
"""

WINDOW_FILTERS = {
    15: "hr = 9 AND mn < 45",
    30: "hr = 9",
    45: "hr = 9 OR (hr = 10 AND mn < 15)",
    60: "hr = 9 OR (hr = 10 AND mn < 30)",
}


def label_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply labeling logic to session data.

    Input DataFrame must have columns:
        session_date, session_high, session_low, w_high, w_low

    Returns DataFrame with added columns:
        label, label_name, binary_label, directional_label, directional_name
    """
    w_high = df["w_high"]
    w_low = df["w_low"]
    session_high = df["session_high"]
    session_low = df["session_low"]

    # Check if window extreme is also session extreme
    w_high_is_session_high = np.isclose(w_high, session_high, rtol=1e-6)
    w_low_is_session_low = np.isclose(w_low, session_low, rtol=1e-6)

    # For trend detection: how far did price extend beyond window range?
    w_range = w_high - w_low
    upside_ext = (session_high - w_high) / w_range.replace(0, float("nan"))
    downside_ext = (w_low - session_low) / w_range.replace(0, float("nan"))
    max_ext = upside_ext.combine(downside_ext, max)

    # Apply labeling conditions (order matters - first match wins)
    conditions = [
        w_high_is_session_high & w_low_is_session_low,   # 3 = double_sweep
        w_high_is_session_high | w_low_is_session_low,   # 2 = reversal
        max_ext <= 0.75,                                  # 1 = containment
    ]
    df["label"] = np.select(conditions, [3, 2, 1], default=0)
    df["label_name"] = df["label"].map(LABEL_NAMES)

    # Binary: reversal or double_sweep vs trend/containment
    df["binary_label"] = df["label"].isin([2, 3]).astype(int)

    # Directional: only meaningful for pure reversals (label=2)
    directional = pd.Series(np.nan, index=df.index)
    directional[w_high_is_session_high & ~w_low_is_session_low] = 1  # fade_the_high
    directional[w_low_is_session_low & ~w_high_is_session_high] = 0  # buy_the_dip
    df["directional_label"] = directional
    df["directional_name"] = directional.map(DIRECTIONAL_NAMES)

    return df


def write_to_clickhouse(ch_client, df: pd.DataFrame, symbol: str, window: int):
    """
    Write label data to the session_labels table in ClickHouse.
    Safe to run multiple times -- ReplacingMergeTree deduplicates on
    (symbol, session_date, window_minutes).
    """
    # Prepare rows for insertion
    rows = []
    for _, row in df.iterrows():
        # Handle NaN in directional_label -> None
        directional = row["directional_label"]
        if pd.isna(directional):
            directional = None
        else:
            directional = int(directional)

        # Determine fh_high_is_session_high and fh_low_is_session_low
        fh_high_is_session_high = int(np.isclose(
            row["w_high"], row["session_high"], rtol=1e-6
        ))
        fh_low_is_session_low = int(np.isclose(
            row["w_low"], row["session_low"], rtol=1e-6
        ))

        rows.append([
            symbol,
            pd.to_datetime(row["session_date"]).date(),
            int(row["label"]),
            row["label_name"],
            int(row["binary_label"]),
            directional,
            fh_high_is_session_high,
            fh_low_is_session_low,
            window,
        ])

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

    ch_client.insert(
        table="session_labels",
        data=rows,
        column_names=columns,
    )

    log.info(f"Written {len(rows)} label rows for {symbol} to session_labels")


def compute_and_write_label(
    ch_client, symbol: str, session_date: date, window: int = 60
) -> None:
    """
    Compute and write a single session label to ClickHouse.

    Used by the scorer's startup health check to backfill missing labels.
    """
    if window not in VALID_WINDOWS:
        raise ValueError(f"Invalid window: {window}. Must be one of {VALID_WINDOWS}")

    window_filter = WINDOW_FILTERS[window]
    sql = SESSION_SQL.format(window_filter=window_filter)

    result = ch_client.query(sql, parameters={
        "symbol": symbol,
        "start": session_date.isoformat(),
        "end": session_date.isoformat(),
    })

    if not result.result_rows:
        log.warning(f"No bar data for {symbol} on {session_date}")
        return

    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    df = label_sessions(df)
    write_to_clickhouse(ch_client, df, symbol, window)


def print_distribution(df: pd.DataFrame, window: int):
    """Print label distribution summary."""
    print(f"\n{'='*60}")
    print(f"Label Distribution (window={window}min, n={len(df)})")
    print(f"{'='*60}")

    print("\nSession Type Distribution:")
    for label in [3, 2, 1, 0]:
        name = LABEL_NAMES[label]
        count = (df["label"] == label).sum()
        pct = 100 * count / len(df)
        print(f"  {name:15} {count:6}  ({pct:5.1f}%)")

    # Binary
    reversal_rate = 100 * df["binary_label"].mean()
    print(f"\nBinary (reversal+double_sweep): {reversal_rate:.1f}%")

    # Directional (only for reversals)
    reversals = df[df["label"] == 2]
    if len(reversals) > 0:
        buy_dip = (reversals["directional_label"] == 0).sum()
        fade_high = (reversals["directional_label"] == 1).sum()
        print(f"\nDirectional (reversals only, n={len(reversals)}):")
        print(f"  buy_the_dip    {buy_dip:6}  ({100*buy_dip/len(reversals):5.1f}%)")
        print(f"  fade_the_high  {fade_high:6}  ({100*fade_high/len(reversals):5.1f}%)")

    # By day of week
    if "session_date" in df.columns:
        df["dow"] = pd.to_datetime(df["session_date"]).dt.dayofweek
        dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
        print("\nReversal rate by day of week:")
        for dow in range(5):
            subset = df[df["dow"] == dow]
            if len(subset) > 0:
                rate = 100 * subset["binary_label"].mean()
                print(f"  {dow_names[dow]}  {rate:5.1f}%  (n={len(subset)})")


def main():
    parser = argparse.ArgumentParser(description="Label trading sessions")
    parser.add_argument("--symbol", required=True, help="Ticker symbol")
    parser.add_argument("--start", default="2021-01-01", help="Start date")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date")
    parser.add_argument("--window", type=int, default=60, choices=VALID_WINDOWS,
                        help="Prediction window in minutes (default: 60)")
    parser.add_argument("--output", "-o", help="Output CSV path (optional)")
    parser.add_argument("--write-db", action="store_true",
                        help="Write labels to ClickHouse session_labels table")
    args = parser.parse_args()

    # Discover ClickHouse
    log.info("Discovering ClickHouse...")
    ch_endpoint = ServiceLocator.wait_for_service(
        ServiceLocator.CLICKHOUSE,
        timeout_sec=60,
    )

    ch_client = clickhouse_connect.get_client(
        host=ch_endpoint.host,
        port=ch_endpoint.port,
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", "Aector99"),
        database=os.getenv("CLICKHOUSE_DATABASE", "trading"),
    )

    # Fetch session data
    window_filter = WINDOW_FILTERS[args.window]
    sql = SESSION_SQL.format(window_filter=window_filter)

    log.info(f"Fetching {args.symbol} sessions ({args.start} -> {args.end})")
    result = ch_client.query(sql, parameters={
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
    })

    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    log.info(f"  {len(df)} sessions retrieved")

    if len(df) == 0:
        log.warning("No sessions found")
        return

    # Apply labels
    df = label_sessions(df)

    # Print distribution
    print_distribution(df, args.window)

    # Write to ClickHouse if requested
    if args.write_db:
        write_to_clickhouse(ch_client, df, args.symbol.upper(), args.window)

    # Output CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        log.info(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
