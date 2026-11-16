"""The SQL symbol parse must match greeks.occ.parse_occ. This runs the actual parse expressions
through ClickHouse on literal symbols (a read-only SELECT, no table access) and compares to
parse_occ. It skips with a clear message when no client is available (no CI here — item 1)."""

from __future__ import annotations

import os
from typing import Any

import pytest

from greeks.occ import parse_occ
from iv_term.term import symbol_parse_exprs

# A call, a put, a fractional strike, and a multi-letter root.
SAMPLES = [
    "AAPL260116C00150000",   # call, 2026-01-16, 150.0, 4-letter root
    "SPY251219P00450000",    # put, 2025-12-19, 450.0, 3-letter root
    "NVDA260417P00180500",   # fractional strike 180.5
    "GOOGL270115C01250000",  # 5-letter root, 1250.0
]


def _client() -> Any:
    if not os.environ.get("CLICKHOUSE_PASSWORD"):
        pytest.skip("no CLICKHOUSE_PASSWORD — skipping the live SQL-parse check (no client)")
    try:
        from ml.shared.clickhouse import get_ch_client
        return get_ch_client()
    except Exception as exc:  # noqa: BLE001 — any failure to build a client is a skip, not a failure
        pytest.skip(f"no ClickHouse client available: {exc}")


def test_sql_symbol_parse_matches_parse_occ() -> None:
    ch = _client()
    for sym in SAMPLES:
        e = symbol_parse_exprs("'" + sym + "'")
        row = ch.query(f"SELECT {e['expiry']} AS expiry, {e['strike']} AS strike, {e['right']} AS right_") \
                .result_rows[0]
        occ = parse_occ(sym)
        assert row[0] == occ.expiry, f"{sym}: expiry {row[0]} != {occ.expiry}"
        assert abs(float(row[1]) - occ.strike) < 1e-9, f"{sym}: strike {row[1]} != {occ.strike}"
        assert str(row[2]) == occ.right.value, f"{sym}: right {row[2]} != {occ.right.value}"
