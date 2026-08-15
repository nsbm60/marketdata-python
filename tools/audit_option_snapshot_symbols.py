#!/usr/bin/env python3
"""Audit option_snapshot symbol quality in ClickHouse.

Runs a set of diagnostic queries to show whether non-OSI symbols exist,
how many rows they affect, when they appeared, and whether they join
option_contract. Per schema, every symbol MUST be OSI — any failure here
is a data-integrity problem, not expected variance.

Usage:
    python tools/audit_option_snapshot_symbols.py
    CLICKHOUSE_HOST=192.168.37.163 python tools/audit_option_snapshot_symbols.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import clickhouse_connect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from discovery.service_locator import ServiceLocator

DATABASE = os.environ.get("CLICKHOUSE_DATABASE", "trading")
SNAPSHOT = "option_snapshot"
CONTRACT = "option_contract"

# Standard OSI: ROOT(1-6 letters) + YYMMDD + C|P + strike*1000 (8 digits)
OSI_RE = r"^[A-Z]{1,6}[0-9]{6}[CP][0-9]{8}$"


def connect_clickhouse():
    host = os.environ.get("CLICKHOUSE_HOST")
    port = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
    if host:
        print(f"ClickHouse (explicit) {host}:{port}\n")
    else:
        ep = ServiceLocator.wait_for_service(ServiceLocator.CLICKHOUSE, timeout_sec=30)
        host, port = ep.host, ep.port
        print(f"ClickHouse discovered at {host}:{port}\n")
    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", "Aector99"),
        database=DATABASE,
    )


def section(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def run(client, title: str, sql: str) -> None:
    section(title)
    result = client.query(sql)
    cols = result.column_names
    widths = [max(len(c), *(len(str(row[i])) for row in result.result_rows))
              for i, c in enumerate(cols)] if result.result_rows else [len(c) for c in cols]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*cols))
    print("  ".join("-" * w for w in widths))
    for row in result.result_rows:
        print(fmt.format(*row))
    if not result.result_rows:
        print("(no rows)")
    print()


def main() -> int:
    ch = connect_clickhouse()
    s, c = SNAPSHOT, CONTRACT

    run(ch, "1) Table scale", f"""
        SELECT
            count()                          AS total_rows,
            uniqExact(symbol)                AS distinct_symbols,
            uniqExact(underlying)            AS distinct_underlyings,
            min(toDate(timestamp))           AS first_day,
            max(toDate(timestamp))           AS last_day
        FROM {DATABASE}.{s}
    """)

    run(ch, "2) OSI compliance (should be 100% — anything else is wrong)", f"""
        SELECT
            countIf(match(symbol, '{OSI_RE}'))                          AS osi_rows,
            countIf(NOT match(symbol, '{OSI_RE}'))                      AS non_osi_rows,
            uniqExactIf(symbol, match(symbol, '{OSI_RE}'))            AS osi_symbols,
            uniqExactIf(symbol, NOT match(symbol, '{OSI_RE}'))        AS non_osi_symbols,
            round(100.0 * countIf(NOT match(symbol, '{OSI_RE}')) / count(), 4)
                                                                        AS non_osi_row_pct
        FROM {DATABASE}.{s}
    """)

    run(ch, "3) Top non-OSI symbols by row count", f"""
        SELECT
            symbol,
            any(underlying)                  AS underlying,
            count()                          AS rows,
            min(toDate(timestamp))           AS first_seen,
            max(toDate(timestamp))           AS last_seen,
            -- tail often shows ZMQ topic suffix: SIDE.STRIKE (e.g. P.705_00)
            substring(symbol, greatest(1, length(symbol) - 11)) AS tail_12
        FROM {DATABASE}.{s}
        WHERE NOT match(symbol, '{OSI_RE}')
        GROUP BY symbol
        ORDER BY rows DESC
        LIMIT 30
    """)

    run(ch, "4) Non-OSI rows by calendar day (when did bad data appear?)", f"""
        SELECT
            toDate(timestamp)                AS d,
            count()                          AS non_osi_rows,
            uniqExact(symbol)                AS non_osi_symbols
        FROM {DATABASE}.{s}
        WHERE NOT match(symbol, '{OSI_RE}')
        GROUP BY d
        ORDER BY d DESC
        LIMIT 40
    """)

    run(ch, "5) Snapshot symbols with NO matching option_contract row", f"""
        SELECT
            count()                          AS orphan_rows,
            uniqExact(s.symbol)              AS orphan_symbols,
            round(100.0 * count() / (SELECT count() FROM {DATABASE}.{s}), 4)
                                             AS orphan_row_pct
        FROM {DATABASE}.{s} AS s
        LEFT JOIN {DATABASE}.{c} AS c ON s.symbol = c.option_symbol
        WHERE c.option_symbol IS NULL
    """)

    run(ch, "6) Top orphan symbols (in snapshot, not in option_contract)", f"""
        SELECT
            s.symbol,
            any(s.underlying)                AS underlying,
            count()                          AS rows,
            match(s.symbol, '{OSI_RE}')      AS looks_like_osi,
            min(toDate(s.timestamp))         AS first_seen,
            max(toDate(s.timestamp))         AS last_seen
        FROM {DATABASE}.{s} AS s
        LEFT JOIN {DATABASE}.{c} AS c ON s.symbol = c.option_symbol
        WHERE c.option_symbol IS NULL
        GROUP BY s.symbol
        ORDER BY rows DESC
        LIMIT 30
    """)

    run(ch, "7) underlying column vs symbol root mismatch (OSI rows only)", f"""
        SELECT
            count()                          AS mismatch_rows,
            uniqExact(symbol)                AS mismatch_symbols
        FROM {DATABASE}.{s}
        WHERE match(symbol, '{OSI_RE}')
          AND underlying != substring(symbol, 1, length(symbol) - 15)
    """)

    run(ch, "8) Sample mismatch examples", f"""
        SELECT
            symbol,
            underlying                       AS stored_underlying,
            substring(symbol, 1, length(symbol) - 15) AS symbol_root,
            count()                          AS rows
        FROM {DATABASE}.{s}
        WHERE match(symbol, '{OSI_RE}')
          AND underlying != substring(symbol, 1, length(symbol) - 15)
        GROUP BY symbol, underlying
        ORDER BY rows DESC
        LIMIT 20
    """)

    run(ch, "9) Symbols containing '.' or '_' (common in non-OSI vendor formats)", f"""
        SELECT
            count()                          AS rows,
            uniqExact(symbol)                AS symbols
        FROM {DATABASE}.{s}
        WHERE position(symbol, '.') > 0 OR position(symbol, '_') > 0
    """)

    run(ch, "10) Top dotted/underscored symbols", f"""
        SELECT
            symbol,
            any(underlying)                  AS underlying,
            count()                          AS rows,
            match(symbol, '{OSI_RE}')        AS is_osi
        FROM {DATABASE}.{s}
        WHERE position(symbol, '.') > 0 OR position(symbol, '_') > 0
        GROUP BY symbol
        ORDER BY rows DESC
        LIMIT 30
    """)

    section("Summary")
    r = ch.query(f"""
        SELECT countIf(NOT match(symbol, '{OSI_RE}')) AS bad
        FROM {DATABASE}.{s}
    """)
    bad = r.result_rows[0][0]
    if bad == 0:
        print("OK: all option_snapshot symbols match OSI format.")
        return 0
    print(f"PROBLEM: {bad:,} rows have non-OSI symbols — see sections 2-4 above.")
    print("These should not exist per schema; trace the ingest path that wrote them.")
    return 1


if __name__ == "__main__":
    sys.exit(main())