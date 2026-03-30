#!/usr/bin/env python3
"""
test_queries.py

Validates all SQL queries in score_session.py against the real ClickHouse database.

Run this BEFORE market open to catch query issues early, not at 10:30am.

Usage:
    python ml/scoring/test_queries.py
    python ml/scoring/test_queries.py --symbols NVDA AMD
    python ml/scoring/test_queries.py --verbose

Exit codes:
    0 = all queries passed
    1 = one or more queries failed
"""

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import clickhouse_connect

from discovery import ServiceLocator

# Import SQL constants from score_session
from ml.scoring.score_session import (
    HISTORY_SQL,
    PRIOR_CLOSE_SQL,
    QQQ_HISTORY_SQL,
    REGIME_SQL,
    HISTORY_SESSIONS,
)


class QueryTest:
    """Single query test case."""

    def __init__(self, name: str, sql: str, params: dict, expected_columns: list[str],
                 min_rows: int = 1, description: str = ""):
        self.name = name
        self.sql = sql
        self.params = params
        self.expected_columns = expected_columns
        self.min_rows = min_rows
        self.description = description


def build_tests(symbols: list[str]) -> list[QueryTest]:
    """Build test cases for all SQL queries."""
    tests = []

    # Use yesterday as prev_date for PRIOR_CLOSE_SQL
    yesterday = date.today() - timedelta(days=1)
    # Find last weekday if yesterday was weekend
    while yesterday.weekday() >= 5:
        yesterday -= timedelta(days=1)

    for symbol in symbols:
        # HISTORY_SQL
        tests.append(QueryTest(
            name=f"HISTORY_SQL ({symbol})",
            sql=HISTORY_SQL,
            params={"symbol": symbol, "n": HISTORY_SESSIONS},
            expected_columns=["session_date", "close_400", "session_high",
                              "session_low", "session_volume", "open_930"],
            min_rows=20,
            description=f"Fetch {HISTORY_SESSIONS} sessions of OHLCV history for {symbol}",
        ))

        # PRIOR_CLOSE_SQL
        tests.append(QueryTest(
            name=f"PRIOR_CLOSE_SQL ({symbol})",
            sql=PRIOR_CLOSE_SQL,
            params={"symbol": symbol, "prev_date": yesterday.isoformat()},
            expected_columns=["prev_close"],
            min_rows=1,
            description=f"Fetch prior close for {symbol} on {yesterday}",
        ))

        # REGIME_SQL
        tests.append(QueryTest(
            name=f"REGIME_SQL ({symbol})",
            sql=REGIME_SQL,
            params={"symbol": symbol},
            expected_columns=["fh_high_is_session_high", "fh_low_is_session_low", "session_date"],
            min_rows=15,
            description=f"Fetch regime features from session_labels for {symbol}",
        ))

    # QQQ_HISTORY_SQL (only need to run once, not per symbol)
    tests.append(QueryTest(
        name="QQQ_HISTORY_SQL",
        sql=QQQ_HISTORY_SQL,
        params={"n": HISTORY_SESSIONS},
        expected_columns=["session_date", "close"],
        min_rows=20,
        description=f"Fetch {HISTORY_SESSIONS} sessions of QQQ close prices",
    ))

    return tests


def run_test(ch_client, test: QueryTest, verbose: bool = False) -> tuple[bool, str]:
    """
    Run a single query test.

    Returns: (passed: bool, message: str)
    """
    try:
        result = ch_client.query(test.sql, parameters=test.params)
        rows = result.result_rows
        columns = result.column_names

        # Check columns match
        missing_cols = set(test.expected_columns) - set(columns)
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"

        # Check minimum rows
        if len(rows) < test.min_rows:
            return False, f"Got {len(rows)} rows, expected >= {test.min_rows}"

        # Check for null values in first row
        if rows:
            first_row = rows[0]
            null_cols = [col for col, val in zip(columns, first_row) if val is None]
            if null_cols:
                return False, f"NULL values in columns: {null_cols}"

        # Success
        msg = f"{len(rows)} rows"
        if verbose and rows:
            # Show sample of first row
            sample = {col: val for col, val in zip(columns, rows[0])}
            msg += f" | sample: {sample}"

        return True, msg

    except Exception as e:
        return False, f"Query error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Validate SQL queries in score_session.py against ClickHouse"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["NVDA", "AMD"],
        help="Symbols to test (default: NVDA AMD)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show sample data from each query"
    )
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols]

    print("=" * 70)
    print("SQL Query Validation for score_session.py")
    print("=" * 70)
    print(f"Symbols: {', '.join(symbols)}")
    print()

    # Connect to ClickHouse via discovery
    print("Discovering ClickHouse...")
    try:
        ch_endpoint = ServiceLocator.wait_for_service(
            ServiceLocator.CLICKHOUSE, timeout_sec=30
        )
    except TimeoutError:
        print("FAILED: Could not discover ClickHouse within 30s")
        sys.exit(1)

    print(f"Connected: {ch_endpoint.host}:{ch_endpoint.port}")
    print()

    ch_client = clickhouse_connect.get_client(
        host=ch_endpoint.host,
        port=ch_endpoint.port,
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", "Aector99"),
        database=os.environ.get("CLICKHOUSE_DATABASE", "trading"),
    )

    # Build and run tests
    tests = build_tests(symbols)

    passed = 0
    failed = 0

    print("-" * 70)
    for test in tests:
        success, message = run_test(ch_client, test, verbose=args.verbose)

        status = "PASS" if success else "FAIL"
        icon = "✓" if success else "✗"

        print(f"{icon} [{status}] {test.name}")
        if args.verbose and test.description:
            print(f"          {test.description}")
        print(f"          {message}")
        print()

        if success:
            passed += 1
        else:
            failed += 1

    # Summary
    print("-" * 70)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print()

    if failed > 0:
        print("FAILED: Some queries did not pass validation")
        sys.exit(1)
    else:
        print("SUCCESS: All queries validated")
        sys.exit(0)


if __name__ == "__main__":
    main()
