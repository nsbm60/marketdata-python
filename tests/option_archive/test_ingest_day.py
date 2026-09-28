"""Trade flat-file parser + the two ClickHouse inserts. No queue/worker anymore."""

from __future__ import annotations

import gzip
from datetime import date, datetime, timezone
from typing import Any

import pytest

from greeks.pull.alpaca_spot import EquityTradePrint
from greeks.pull.massive_trades import ns_to_utc
from option_archive.domain import to_osi
from option_archive.ingest_day import (
    LedgerRow,
    _attach_spot,
    _build_rows,
    insert_ingest_log,
    insert_option_trades,
    parse_trades,
)

_HEADER = "ticker,conditions,correction,exchange,price,sip_timestamp,size"  # real 7-col file
_C1 = to_osi("NVDA260417C00180000")
_SIP = 1_680_000_000_000_000_000  # ns
_SIP2 = _SIP + 5_000_000_000


def _gz(rows: list[str]) -> bytes:
    return gzip.compress(("\n".join([_HEADER] + rows) + "\n").encode())


# One contract, three prints — first two byte-identical (the dup case), sip-ascending.
_ROWS = [
    f"O:NVDA260417C00180000,232,0,312,11.82,{_SIP},2",
    f"O:NVDA260417C00180000,232,0,312,11.82,{_SIP},2",
    f"O:NVDA260417C00180000,209,0,325,11.07,{_SIP2},1",
]


class _FakeCH:
    def __init__(self) -> None:
        self.inserts: list[Any] = []

    def insert(self, name: str, data: Any, column_names: Any, database: Any) -> None:
        self.inserts.append((name, data, column_names, database))


# -- parse --------------------------------------------------------------------


def test_parse_assigns_ordinals_and_preserves_duplicates() -> None:
    prints = parse_trades(_gz(_ROWS), frozenset({"NVDA"}))[_C1]
    assert [p.ordinal for p in prints] == [0, 1, 2]
    assert prints[0].price == 11.82 and prints[1].price == 11.82  # dup preserved
    assert prints[0].conditions == (232,)
    assert prints[0].correction == 0
    assert prints[0].participant_timestamp_ns is None  # 7-col file omits it
    assert prints[2].exchange == 325


def test_parse_sorts_by_sip_before_ordinal_stable() -> None:
    # File order is sip-DESCENDING with a tie; ordinals must follow sip ascending,
    # and equal timestamps keep file order (stable) — answer #3.
    rows = [
        f"O:NVDA260417C00180000,1,0,1,2.0,{_SIP2},1",  # later sip, first in file
        f"O:NVDA260417C00180000,1,0,1,1.0,{_SIP},1",    # earlier sip
        f"O:NVDA260417C00180000,1,0,1,1.5,{_SIP},1",    # tie -> keeps file order after the 1.0 row
    ]
    prints = parse_trades(_gz(rows), frozenset({"NVDA"}))[_C1]
    assert [p.sip_timestamp_ns for p in prints] == [_SIP, _SIP, _SIP2]
    assert [p.ordinal for p in prints] == [0, 1, 2]
    assert [p.price for p in prints] == [1.0, 1.5, 2.0]  # tie kept file order (1.0 before 1.5)


def test_parse_reads_participant_timestamp_when_present() -> None:
    header = "ticker,conditions,correction,exchange,participant_timestamp,price,sip_timestamp,size"
    gz = gzip.compress(
        (header + f"\nO:NVDA260417C00180000,232,0,312,{_SIP - 500},11.82,{_SIP},2\n").encode()
    )
    p = parse_trades(gz, frozenset({"NVDA"}))[_C1][0]
    assert p.participant_timestamp_ns == _SIP - 500
    assert p.sip_timestamp_ns == _SIP and p.price == 11.82


def test_parse_filters_to_watchlist_root() -> None:
    assert parse_trades(_gz(_ROWS), frozenset({"AAPL"})) == {}  # NVDA root not kept


def test_parse_rejects_missing_required_columns() -> None:
    bad = gzip.compress(b"ticker,price\nO:X,1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        parse_trades(bad, frozenset({"NVDA"}))


# -- inserts ------------------------------------------------------------------


def test_insert_option_trades_shapes_rows_with_spot() -> None:
    by = parse_trades(_gz(_ROWS), frozenset({"NVDA"}))
    tape = [EquityTradePrint("NVDA", ns_to_utc(_SIP - 1_000_000), 180.0, 100.0)]
    spot = _attach_spot(sorted(by[_C1], key=lambda p: p.trade_ts), tape)
    rows = _build_rows(by, spot, date(2026, 4, 1))
    ch = _FakeCH()
    assert insert_option_trades(ch, rows, table="trading.option_trade") == 3
    name, data, cols, db = ch.inserts[0]
    assert name == "option_trade" and db == "trading"
    assert cols[9] == "ordinal" and {r[9] for r in data} == {0, 1, 2}
    si = cols.index("spot_at_trade")
    assert all(r[si] == 180.0 for r in data)


def test_insert_ingest_log_carries_enumeration_misses() -> None:
    ch = _FakeCH()
    now = datetime(2026, 4, 2, tzinfo=timezone.utc)
    row = LedgerRow(
        session_date=date(2026, 4, 1), transport="flatfile", tasks_success=5,
        tasks_no_trades=0, tasks_failed=0, rows_inserted=42, bytes_downloaded=1000,
        wall_seconds=1.5, vendor_volume_delta=None, enumeration_misses=7,
        started_at=now, finished_at=now,
    )
    insert_ingest_log(ch, row, table="trading.ingest_log")
    _name, data, cols, _db = ch.inserts[0]
    d = dict(zip(cols, data[0]))
    assert "enumeration_misses" in cols
    assert d["enumeration_misses"] == 7 and d["rows_inserted"] == 42
