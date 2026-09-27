"""Per-day trade-file ingest (PR3). S3, Alpaca, and ClickHouse are faked."""

from __future__ import annotations

import gzip
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from greeks.pull.alpaca_spot import EquityTradePrint
from greeks.pull.massive_trades import ns_to_utc
from option_archive import ingest_day as mod
from option_archive.config import get_config
from option_archive.domain import TaskStatus, to_osi
from option_archive.ingest_day import ingest_day, parse_trades
from option_archive.queue import WorkQueue

_HEADER = "ticker,conditions,correction,exchange,price,sip_timestamp,size"  # real 7-col file
_C1 = to_osi("NVDA260417C00180000")
_C2 = to_osi("NVDA260417P00180000")  # enumerated but absent from the file
_D = date(2026, 4, 1)
_SIP = 1_680_000_000_000_000_000  # ns
_SIP2 = _SIP + 5_000_000_000
_NOW = datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)


def _gz(rows: list[str]) -> bytes:
    return gzip.compress(("\n".join([_HEADER] + rows) + "\n").encode())


# One contract, three prints — the first two byte-identical (the dup case).
_ROWS = [
    f"O:NVDA260417C00180000,232,0,312,11.82,{_SIP},2",
    f"O:NVDA260417C00180000,232,0,312,11.82,{_SIP},2",
    f"O:NVDA260417C00180000,209,0,325,11.07,{_SIP2},1",
]


class _Body:
    def __init__(self, b: bytes) -> None:
        self._b = b

    def read(self) -> bytes:
        return self._b


class _FakeS3:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def get_object(self, Bucket: str, Key: str) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return {"Body": _Body(self._payload)}


class _FakeCH:
    def __init__(self) -> None:
        self.inserts: list[Any] = []

    def insert(self, name: str, data: Any, column_names: Any, database: Any) -> None:
        self.inserts.append((name, data, column_names, database))


def _queue(tmp_path: Path) -> WorkQueue:
    return WorkQueue(
        tmp_path / "q.db", lease=timedelta(minutes=30), max_attempts=5,
        backoff_base=timedelta(seconds=60),
    )


def _tape(*_a: Any, **_k: Any) -> list[EquityTradePrint]:
    # one equity print just before the option trades -> spot attaches to all
    return [EquityTradePrint("NVDA", ns_to_utc(_SIP - 1_000_000), 180.0, 100.0)]


# -- parse --------------------------------------------------------------------


def test_parse_assigns_ordinals_and_preserves_duplicates() -> None:
    by_sym = parse_trades(_gz(_ROWS), frozenset({_C1}))
    prints = by_sym[_C1]
    assert [p.ordinal for p in prints] == [0, 1, 2]  # stable file order
    assert prints[0].price == 11.82 and prints[1].price == 11.82  # dup preserved as two rows
    assert prints[0].conditions == (232,)
    assert prints[0].correction == 0
    assert prints[0].participant_timestamp_ns is None  # 7-col file omits it
    assert prints[2].exchange == 325


def test_parse_reads_participant_timestamp_when_present() -> None:
    # The documented 8-col layout: participant_timestamp populated, parsed by name.
    header = "ticker,conditions,correction,exchange,participant_timestamp,price,sip_timestamp,size"
    gz = gzip.compress(
        (header + f"\nO:NVDA260417C00180000,232,0,312,{_SIP - 500},11.82,{_SIP},2\n").encode()
    )
    p = parse_trades(gz, frozenset({_C1}))[_C1][0]
    assert p.participant_timestamp_ns == _SIP - 500
    assert p.sip_timestamp_ns == _SIP and p.price == 11.82


def test_parse_filters_to_keep_set() -> None:
    assert parse_trades(_gz(_ROWS), frozenset({_C2})) == {}  # C1 not kept


def test_parse_rejects_missing_required_columns() -> None:
    bad = gzip.compress(b"ticker,price\nO:X,1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        parse_trades(bad, frozenset({_C1}))


# -- ingest_day ---------------------------------------------------------------


def test_ingest_day_inserts_and_marks_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "equity_tape", _tape)
    q = _queue(tmp_path)
    q.enqueue_many([(_C1, _D), (_C2, _D)])
    ch = _FakeCH()

    rep = ingest_day(q, ch, _FakeS3(_gz(_ROWS)), object(), get_config(), now=_NOW)

    assert rep is not None and rep.outcome == "done"
    assert rep.contracts_claimed == 2
    assert rep.trades_inserted == 3          # dups preserved
    assert rep.no_trade_contracts == 1       # C2 absent from the file
    # one bulk insert of 3 rows, spot attached
    (_name, data, cols, _db) = ch.inserts[0]
    assert len(data) == 3
    assert cols[9] == "ordinal"
    assert {row[9] for row in data} == {0, 1, 2}
    spot_idx = cols.index("spot_at_trade")
    assert all(row[spot_idx] == 180.0 for row in data)
    # both jobs terminal DONE (C2 = no-trades = DONE + zero rows)
    assert q.counts() == {TaskStatus.DONE: 2}
    # ledger row written as the final act, after mark_done
    (_ln, ldata, lcols, _ldb) = next(i for i in ch.inserts if i[0] == "ingest_log")
    lrow = dict(zip(lcols, ldata[0]))
    assert lrow["transport"] == "flatfile"
    assert lrow["tasks_success"] == 1 and lrow["tasks_no_trades"] == 1
    assert lrow["tasks_failed"] == 0 and lrow["rows_inserted"] == 3
    assert lrow["vendor_volume_delta"] is None


def test_ingest_day_s3_failure_releases_without_penalty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "equity_tape", _tape)
    q = _queue(tmp_path)
    q.enqueue_many([(_C1, _D), (_C2, _D)])
    err = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    rep = ingest_day(q, _FakeCH(), _FakeS3(err), object(), get_config(), now=_NOW)

    assert rep is not None and rep.outcome == "transport"
    assert q.counts() == {TaskStatus.PENDING: 2}  # back to pending, no FAILED


def test_ingest_day_none_when_drained(tmp_path: Path) -> None:
    assert ingest_day(_queue(tmp_path), _FakeCH(), _FakeS3(_gz(_ROWS)), object(),
                      get_config(), now=_NOW) is None
