"""
alpaca_bars_etl.py

Pull historical 1-minute OHLCV bars from Alpaca and insert into ClickHouse.

Usage:
    python alpaca_bars_etl.py --symbol NVDA
    python alpaca_bars_etl.py --symbol NVDA --start 2023-01-01 --end 2024-12-31
    python alpaca_bars_etl.py --symbol AMD --start 2023-06-01

Dependencies:
    pip install alpaca-py clickhouse-connect python-dotenv pyzmq

Environment variables (in .env or shell):
    ALPACA_API_KEY
    ALPACA_API_SECRET
    CLICKHOUSE_USER       (default: default)
    CLICKHOUSE_PASSWORD   (default: Aector99)
    CLICKHOUSE_DATABASE   (default: trading)

ClickHouse host/port are discovered automatically via ZMQ discovery bus.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, date, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Add parent directory to path for discovery module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import clickhouse_connect
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed
from dotenv import load_dotenv

from discovery import ServiceLocator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")

# Regular session bounds (NY time)
REGULAR_START_HOUR = 9
REGULAR_START_MINUTE = 30
REGULAR_END_HOUR = 16
REGULAR_END_MINUTE = 0

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stock_bars_1m (
    symbol      LowCardinality(String),
    ts          DateTime64(3, 'America/New_York'),
    open        Float64,
    high        Float64,
    low         Float64,
    close       Float64,
    volume      UInt64,
    vwap        Nullable(Float64),
    trade_count Nullable(UInt32),
    session     Enum8('pre' = 0, 'regular' = 1, 'post' = 2)
) ENGINE = ReplacingMergeTree()
ORDER BY (symbol, ts);
"""


def classify_session(ts: datetime) -> int:
    """Return session enum value for a NY-local timestamp."""
    t = ts.astimezone(NY).time()
    regular_open  = t.replace(hour=REGULAR_START_HOUR, minute=REGULAR_START_MINUTE, second=0, microsecond=0)
    regular_close = t.replace(hour=REGULAR_END_HOUR,   minute=REGULAR_END_MINUTE,   second=0, microsecond=0)
    # compare time objects directly
    from datetime import time as dtime
    ro = dtime(REGULAR_START_HOUR, REGULAR_START_MINUTE)
    rc = dtime(REGULAR_END_HOUR,   REGULAR_END_MINUTE)
    if ro <= t < rc:
        return 1   # regular
    elif t < ro:
        return 0   # pre
    else:
        return 2   # post


def get_last_stored_ts(ch_client, symbol: str) -> datetime | None:
    """Return the most recent timestamp already stored for this symbol, or None."""
    result = ch_client.query(
        "SELECT max(ts) FROM stock_bars_1m WHERE symbol = %(symbol)s",
        parameters={"symbol": symbol},
    )
    val = result.first_item.get("max(ts)")
    if val is None:
        return None
    # clickhouse-connect returns datetime already
    if isinstance(val, datetime):
        return val.replace(tzinfo=timezone.utc)
    return None


def pull_and_insert(
    symbol: str,
    start: date,
    end: date,
    alpaca_client: StockHistoricalDataClient,
    ch_client,
    batch_size: int = 10_000,
) -> int:
    """Pull bars from Alpaca and insert into ClickHouse. Returns total rows inserted."""

    log.info(f"Pulling {symbol} bars from {start} to {end}")

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
        end=datetime(end.year, end.month, end.day, 23, 59, tzinfo=timezone.utc),
        adjustment=Adjustment.ALL,
        feed=DataFeed.SIP,
        extended_hours=True,
    )

    log.info("Fetching from Alpaca (this may take a moment for large ranges)...")
    bars_response = alpaca_client.get_stock_bars(request)

    # bars_response[symbol] is a list of Bar objects
    bars = bars_response[symbol] if symbol in bars_response.data else []

    if not bars:
        log.warning(f"No bars returned for {symbol}")
        return 0

    log.info(f"Received {len(bars):,} bars from Alpaca, inserting into ClickHouse...")

    total_inserted = 0
    batch = []

    for bar in bars:
        ts = bar.timestamp
        if not isinstance(ts, datetime):
            ts = datetime.fromisoformat(str(ts))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        batch.append([
            symbol,
            ts,
            float(bar.open),
            float(bar.high),
            float(bar.low),
            float(bar.close),
            int(bar.volume),
            float(bar.vwap)        if bar.vwap        is not None else None,
            int(bar.trade_count)   if bar.trade_count is not None else None,
            classify_session(ts),
        ])

        if len(batch) >= batch_size:
            _insert_batch(ch_client, batch)
            total_inserted += len(batch)
            log.info(f"  {total_inserted:>10,} rows inserted...")
            batch = []

    if batch:
        _insert_batch(ch_client, batch)
        total_inserted += len(batch)

    log.info(f"Insert complete: {total_inserted:,} rows for {symbol}")
    return total_inserted


def _insert_batch(ch_client, batch: list) -> None:
    ch_client.insert(
        "stock_bars_1m",
        batch,
        column_names=[
            "symbol", "ts", "open", "high", "low", "close",
            "volume", "vwap", "trade_count", "session",
        ],
    )


def validate(ch_client, symbol: str, start: date, end: date) -> None:
    """Print a quick sanity-check summary after the pull."""
    log.info("--- Validation summary ---")

    result = ch_client.query(
        """
        SELECT
            count()                                         AS total_bars,
            countIf(session = 1)                            AS regular_bars,
            countIf(session = 0)                            AS pre_bars,
            countIf(session = 2)                            AS post_bars,
            countIf(session = 1) / count(DISTINCT toDate(ts)) AS avg_regular_per_day,
            min(ts)                                         AS earliest,
            max(ts)                                         AS latest
        FROM stock_bars_1m
        WHERE symbol = %(symbol)s
          AND toDate(ts) BETWEEN %(start)s AND %(end)s
        """,
        parameters={"symbol": symbol, "start": str(start), "end": str(end)},
    )
    row = result.first_item
    log.info(f"  Total bars:            {row['total_bars']:,}")
    log.info(f"  Regular session bars:  {row['regular_bars']:,}")
    log.info(f"  Pre-market bars:       {row['pre_bars']:,}")
    log.info(f"  Post-market bars:      {row['post_bars']:,}")
    log.info(f"  Avg regular/day:       {row['avg_regular_per_day']:.1f}  (expect ~390)")
    log.info(f"  Date range stored:     {row['earliest']}  →  {row['latest']}")

    # Flag sessions with abnormally low regular bar counts (halts / data gaps)
    gap_result = ch_client.query(
        """
        SELECT toDate(ts) AS session_date, countIf(session = 1) AS regular_count
        FROM stock_bars_1m
        WHERE symbol = %(symbol)s
          AND toDate(ts) BETWEEN %(start)s AND %(end)s
          AND toDayOfWeek(ts) NOT IN (6, 7)   -- exclude weekends
        GROUP BY session_date
        HAVING regular_count < 300
        ORDER BY session_date
        """,
        parameters={"symbol": symbol, "start": str(start), "end": str(end)},
    )
    gap_rows = gap_result.result_rows
    if gap_rows:
        log.warning(f"  Sessions with <300 regular bars (halts / gaps): {len(gap_rows)}")
        for r in gap_rows[:10]:
            log.warning(f"    {r[0]}  {r[1]} bars")
        if len(gap_rows) > 10:
            log.warning(f"    ... and {len(gap_rows) - 10} more")
    else:
        log.info("  No anomalous sessions detected.")


def main():
    parser = argparse.ArgumentParser(description="Pull Alpaca 1-min bars into ClickHouse")
    parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g. NVDA")
    parser.add_argument("--start",  default="2023-01-01", help="Start date YYYY-MM-DD (default: 2023-01-01)")
    parser.add_argument("--end",    default=date.today().isoformat(), help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    start  = date.fromisoformat(args.start)
    end    = date.fromisoformat(args.end)

    if start >= end:
        log.error("--start must be before --end")
        sys.exit(1)

    # --- Alpaca client ---
    api_key    = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        log.error("ALPACA_API_KEY and ALPACA_API_SECRET must be set")
        sys.exit(1)

    alpaca_client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)

    # --- ClickHouse client (discovered via ZMQ) ---
    log.info("Discovering ClickHouse...")
    ch_endpoint = ServiceLocator.wait_for_service(
        ServiceLocator.CLICKHOUSE,
        timeout_sec=60,
    )

    ch_client = clickhouse_connect.get_client(
        host     = ch_endpoint.host,
        port     = ch_endpoint.port,
        username = os.environ.get("CLICKHOUSE_USER",     "default"),
        password = os.environ.get("CLICKHOUSE_PASSWORD", "Aector99"),
        database = os.environ.get("CLICKHOUSE_DATABASE", "trading"),
    )

    # Ensure table exists
    ch_client.command(CREATE_TABLE_SQL)
    log.info("Table stock_bars_1m ready")

    # Resumption: advance start date if data already exists
    last_ts = get_last_stored_ts(ch_client, symbol)
    if last_ts is not None:
        last_date = last_ts.astimezone(NY).date()
        if last_date >= end:
            log.info(f"{symbol} already loaded through {last_date}, nothing to do")
            sys.exit(0)
        if last_date > start:
            log.info(f"Resuming from {last_date} (last stored date for {symbol})")
            start = last_date  # re-pull last day to catch any missing bars

    total = pull_and_insert(symbol, start, end, alpaca_client, ch_client)

    if total > 0:
        validate(ch_client, symbol, start, end)

    ch_client.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
