#!/usr/bin/env python3
"""
migrate_stock_bars.py

Migrates 1m bar history from stock_bar_1m (ET timezone) to stock_bar
(UTC, with period column). After migration, verifies completeness per
symbol per date and fetches any missing bars from Alpaca.

Usage:
    python tools/migrate_stock_bars.py [--dry-run]
    python tools/migrate_stock_bars.py [--symbols NVDA,AMD,...]
    python tools/migrate_stock_bars.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]

Steps:
    1. Discover ClickHouse via ZMQ discovery bus
    2. Migrate all rows from stock_bar_1m → stock_bar (period='1m', ts in UTC)
    3. Verify counts per symbol per date match between old and new table
    4. For any gaps, fetch missing bars from Alpaca and insert into stock_bar
    5. Final verification report

Requirements:
    pip install clickhouse-connect requests

Environment variables:
    ALPACA_API_KEY, ALPACA_API_SECRET
    CLICKHOUSE_USER     (default: default)
    CLICKHOUSE_PASSWORD (default: Aector99)
    CLICKHOUSE_DATABASE (default: trading)
"""

import logging
import os
import sys
import time
import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import requests
import clickhouse_connect
from dotenv import load_dotenv

# Add project root to path for discovery module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from discovery.service_locator import ServiceLocator

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Migrate stock_bar_1m to stock_bar")
parser.add_argument("--symbols",    default=None, help="Comma-separated symbols (default: all)")
parser.add_argument("--dry-run",    action="store_true", help="Report only, no writes")
parser.add_argument("--start-date", default=None, help="YYYY-MM-DD — only migrate from this date")
parser.add_argument("--end-date",   default=None, help="YYYY-MM-DD — only migrate up to this date")
args = parser.parse_args()

API_KEY    = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_API_SECRET")

if not API_KEY or not API_SECRET:
    log.error("Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
    sys.exit(1)

DRY_RUN = args.dry_run
if DRY_RUN:
    log.info("*** DRY RUN MODE — no writes will be made ***")

# ---------------------------------------------------------------------------
# Step 1: Discover ClickHouse
# ---------------------------------------------------------------------------
log.info("=" * 60)
log.info("Step 1: Discovering ClickHouse")
log.info("=" * 60)

endpoint = ServiceLocator.wait_for_service(
    service_name=ServiceLocator.CLICKHOUSE,
    timeout_sec=30,
)
log.info(f"ClickHouse discovered: {endpoint.host}:{endpoint.port}")

ch = clickhouse_connect.get_client(
    host     = endpoint.host,
    port     = endpoint.port,
    username = os.environ.get("CLICKHOUSE_USER",     "default"),
    password = os.environ.get("CLICKHOUSE_PASSWORD", "Aector99"),
    database = os.environ.get("CLICKHOUSE_DATABASE", "trading"),
)

# ---------------------------------------------------------------------------
# Step 2: Determine symbols to migrate
# ---------------------------------------------------------------------------
log.info("=" * 60)
log.info("Step 2: Determining symbols")
log.info("=" * 60)

if args.symbols:
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    log.info(f"Using provided symbols: {symbols}")
else:
    result = ch.query("SELECT DISTINCT symbol FROM stock_bar_1m ORDER BY symbol")
    symbols = [r[0] for r in result.result_rows]
    log.info(f"Found {len(symbols)} symbols in stock_bar_1m: {symbols}")

# ---------------------------------------------------------------------------
# Build filters
# ---------------------------------------------------------------------------
symbol_list   = ", ".join(f"'{s}'" for s in symbols)
symbol_filter = f"AND symbol IN ({symbol_list})"

date_filter = ""
if args.start_date:
    date_filter += f" AND toDate(toTimezone(ts, 'UTC')) >= '{args.start_date}'"
if args.end_date:
    date_filter += f" AND toDate(toTimezone(ts, 'UTC')) <= '{args.end_date}'"

# ---------------------------------------------------------------------------
# Step 3: Migrate rows from stock_bar_1m → stock_bar
# ---------------------------------------------------------------------------
log.info("=" * 60)
log.info("Step 3: Migrating rows")
log.info("=" * 60)

total_source = ch.query(f"""
    SELECT count()
    FROM stock_bar_1m
    WHERE 1=1 {symbol_filter} {date_filter}
""").result_rows[0][0]
log.info(f"Source rows to migrate: {total_source:,}")

if not DRY_RUN:
    log.info("Running migration INSERT...SELECT...")
    start = time.time()
    ch.command(f"""
        INSERT INTO stock_bar (symbol, period, ts, open, high, low, close, volume, vwap, trade_count, session)
        SELECT
            symbol,
            '1m'                                               AS period,
            toDateTime64(toTimeZone(ts, 'UTC'), 3, 'UTC')     AS ts,
            open, high, low, close, volume, vwap, trade_count, session
        FROM stock_bar_1m
        WHERE 1=1 {symbol_filter} {date_filter}
    """)
    elapsed = time.time() - start
    log.info(f"Migration complete in {elapsed:.1f}s")
else:
    log.info(f"[DRY RUN] Would migrate {total_source:,} rows")

# ---------------------------------------------------------------------------
# Step 4: Verify counts per symbol per date
# ---------------------------------------------------------------------------
log.info("=" * 60)
log.info("Step 4: Verifying counts per symbol per date")
log.info("=" * 60)

# Source counts — use ET date since stock_bar_1m stores timestamps in ET
old_result = ch.query(f"""
    SELECT
        symbol,
        toDate(toTimezone(ts, 'America/New_York')) AS bar_date,
        count()                                    AS cnt
    FROM stock_bar_1m
    WHERE session = 1 {symbol_filter} {date_filter}
    GROUP BY symbol, bar_date
    ORDER BY symbol, bar_date
""")
old_counts = {(r[0], r[1]): r[2] for r in old_result.result_rows}
log.info(f"Source: {len(old_counts):,} symbol/date combinations")

gaps = []
if not DRY_RUN:
    # New counts — stock_bar stores in UTC, convert to ET for date comparison
    new_result = ch.query(f"""
        SELECT
            symbol,
            toDate(toTimezone(ts, 'America/New_York')) AS bar_date,
            count()                                    AS cnt
        FROM stock_bar FINAL
        WHERE period = '1m' AND session = 1 {symbol_filter} {date_filter}
        GROUP BY symbol, bar_date
        ORDER BY symbol, bar_date
    """)
    new_counts = {(r[0], r[1]): r[2] for r in new_result.result_rows}

    for (symbol, bar_date), old_cnt in old_counts.items():
        new_cnt = new_counts.get((symbol, bar_date), 0)
        if new_cnt < old_cnt:
            gaps.append((symbol, bar_date, old_cnt, new_cnt))

    if not gaps:
        log.info(f"✓ All {len(old_counts):,} symbol/date combinations match")
    else:
        log.warning(f"Found {len(gaps)} symbol/date gaps:")
        for symbol, bar_date, old_cnt, new_cnt in gaps[:20]:
            log.warning(f"  {symbol} {bar_date}: source={old_cnt}, migrated={new_cnt}, missing={old_cnt - new_cnt}")
        if len(gaps) > 20:
            log.warning(f"  ... and {len(gaps) - 20} more")
else:
    log.info(f"[DRY RUN] {len(old_counts):,} symbol/date combinations found in source")

# ---------------------------------------------------------------------------
# Step 5: Fetch missing bars from Alpaca
# ---------------------------------------------------------------------------
ALPACA_BASE    = "https://data.alpaca.markets/v2"
ALPACA_HEADERS = {
    "APCA-API-KEY-ID":     API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
}

def fetch_alpaca_bars(symbol: str, bar_date: date) -> list[dict]:
    """Fetch 1m RTH bars for a symbol/date from Alpaca historical API."""
    start_dt = datetime(bar_date.year, bar_date.month, bar_date.day,
                        13, 30, tzinfo=timezone.utc)  # 09:30 ET = 13:30 UTC
    end_dt   = datetime(bar_date.year, bar_date.month, bar_date.day,
                        20, 0,  tzinfo=timezone.utc)  # 16:00 ET = 20:00 UTC

    params = {
        "timeframe": "1Min",
        "start":     start_dt.isoformat(),
        "end":       end_dt.isoformat(),
        "feed":      "sip",
        "limit":     1000,
    }

    bars = []
    url = f"{ALPACA_BASE}/stocks/{symbol}/bars"
    while True:
        resp = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=30)
        if resp.status_code != 200:
            log.error(f"  Alpaca error for {symbol} {bar_date}: {resp.status_code} {resp.text[:200]}")
            return []
        data = resp.json()
        bars.extend(data.get("bars", []))
        next_token = data.get("next_page_token")
        if not next_token:
            break
        params["page_token"] = next_token

    return bars

def insert_alpaca_bars(symbol: str, bars: list[dict]) -> int:
    """Insert Alpaca bar dicts into stock_bar."""
    if not bars:
        return 0

    rows = []
    for b in bars:
        ts_utc = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
        rows.append([
            symbol,
            "1m",
            ts_utc,
            float(b["o"]),
            float(b["h"]),
            float(b["l"]),
            float(b["c"]),
            int(b["v"]),
            float(b.get("vw") or 0.0) or None,
            int(b.get("n") or 0) or None,
            1,  # session = regular
        ])

    ch.insert(
        "stock_bar",
        rows,
        column_names=["symbol", "period", "ts", "open", "high", "low", "close",
                      "volume", "vwap", "trade_count", "session"]
    )
    return len(rows)

if gaps and not DRY_RUN:
    log.info("=" * 60)
    log.info("Step 5: Fetching missing bars from Alpaca")
    log.info("=" * 60)

    filled = 0
    failed = 0
    for symbol, bar_date, old_cnt, new_cnt in gaps:
        missing = old_cnt - new_cnt
        log.info(f"  Fetching {symbol} {bar_date} ({missing} missing bars)...")
        bars = fetch_alpaca_bars(symbol, bar_date)
        if bars:
            inserted = insert_alpaca_bars(symbol, bars)
            log.info(f"    ✓ Inserted {inserted} bars from Alpaca")
            filled += 1
        else:
            log.warning(f"    ✗ No bars returned from Alpaca for {symbol} {bar_date}")
            failed += 1
        time.sleep(0.2)  # rate limit courtesy

    log.info(f"Gap fill complete: {filled} symbol/dates filled, {failed} failed")

else:
    if DRY_RUN:
        log.info("[DRY RUN] Skipping gap fill")

# ---------------------------------------------------------------------------
# Step 6: Final verification report
# ---------------------------------------------------------------------------
log.info("=" * 60)
log.info("Step 6: Final verification")
log.info("=" * 60)

if not DRY_RUN:
    final_result = ch.query(f"""
        SELECT
            symbol,
            toDate(toTimezone(ts, 'America/New_York')) AS bar_date,
            count()                                    AS cnt
        FROM stock_bar FINAL
        WHERE period = '1m' AND session = 1 {symbol_filter} {date_filter}
        GROUP BY symbol, bar_date
        ORDER BY symbol, bar_date
    """)
    final_counts = {(r[0], r[1]): r[2] for r in final_result.result_rows}

    remaining_gaps = []
    for (symbol, bar_date), old_cnt in old_counts.items():
        new_cnt = final_counts.get((symbol, bar_date), 0)
        if new_cnt < old_cnt:
            remaining_gaps.append((symbol, bar_date, old_cnt, new_cnt))

    total_new = ch.query(f"""
        SELECT count()
        FROM stock_bar FINAL
        WHERE period = '1m' {symbol_filter} {date_filter}
    """).result_rows[0][0]

    log.info(f"stock_bar  (period='1m'): {total_new:,} rows")
    log.info(f"stock_bar_1m (source):   {total_source:,} rows")

    if not remaining_gaps:
        log.info(f"✓ Migration complete — all {len(old_counts):,} symbol/date combinations verified")
        log.info("")
        log.info("Next steps:")
        log.info("  1. Deploy updated MDS (all bar queries now target stock_bar)")
        log.info("  2. Verify new 1m and 5m bars flowing into stock_bar correctly")
        log.info("  3. Run: DROP TABLE trading.stock_bar_1m")
    else:
        log.warning(f"{len(remaining_gaps)} symbol/date combinations still have gaps after Alpaca fill:")
        for symbol, bar_date, old_cnt, new_cnt in remaining_gaps:
            log.warning(f"  {symbol} {bar_date}: expected={old_cnt}, actual={new_cnt}")
        log.warning("Investigate before dropping stock_bar_1m.")
else:
    log.info(f"[DRY RUN] Source: {total_source:,} rows across {len(old_counts):,} symbol/date combinations")
