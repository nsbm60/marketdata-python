"""The single-program orchestration: resume, trading-day union, the not-listed
rule, and tempfile cleanup. ClickHouse, S3, and the Massive reference are faked."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from greeks.domain import OptionRight
from greeks.pull.contracts import ContractRef
from option_archive import archive
from option_archive.config import get_config
from option_archive.ingest_day import FlatTradePrint
from option_archive.quotes import QuotePull


class _FakeCH:
    """Captures inserts; answers the resume/done queries from canned rows."""

    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows
        self.inserts: list[tuple] = []

    def query(self, sql: str, parameters: Any = None) -> Any:
        return type("R", (), {"result_rows": self._rows})()

    def insert(self, name: str, data: Any, column_names: Any, database: Any) -> None:
        self.inserts.append((name, data, column_names, database))


# -- resume -------------------------------------------------------------------


def test_resume_empty_ledger_uses_config_floor() -> None:
    cfg = get_config()
    ch = _FakeCH([(0, date(1970, 1, 1))])  # count=0 -> empty ledger
    assert archive._resume_start_date(ch, cfg) == cfg.backfill_start_date


def test_resume_backs_up_overlap_days() -> None:
    cfg = get_config()
    ch = _FakeCH([(42, date(2026, 6, 20))])
    assert archive._resume_start_date(ch, cfg) == date(2026, 6, 20) - timedelta(days=3)


def test_resume_never_before_config_floor() -> None:
    cfg = get_config()
    ch = _FakeCH([(1, cfg.backfill_start_date + timedelta(days=1))])
    # max - 3 days would precede the floor; clamp to the floor
    assert archive._resume_start_date(ch, cfg) == cfg.backfill_start_date


# -- trading days -------------------------------------------------------------


def test_trading_days_union_and_window() -> None:
    spots = {
        "AAA": {date(2026, 1, 5): 1.0, date(2026, 1, 6): 1.0},
        "BBB": {date(2026, 1, 6): 2.0, date(2026, 1, 7): 2.0, date(2025, 12, 31): 2.0},
    }
    days = archive._trading_days(spots, date(2026, 1, 5), date(2026, 1, 6))
    assert days == [date(2026, 1, 5), date(2026, 1, 6)]  # union, clipped to the window


# -- not-listed rule ----------------------------------------------------------


def _osi(expiry: date, strike: float) -> str:
    return f"NVDA{expiry:%y%m%d}C{int(round(strike * 1000)):08d}"


def _cref(osi: str, expiry: date, strike: float, shares: int = 100) -> ContractRef:
    return ContractRef(
        osi=osi, massive_ticker="O:" + osi, underlying="NVDA", expiry=expiry,
        strike=strike, right=OptionRight.CALL, shares_per_contract=shares,
        exercise_style="american", primary_exchange=None,
    )


def _prints(osi: str, n: int) -> list[FlatTradePrint]:
    return [
        FlatTradePrint(osi, i, 1_680_000_000_000_000_000 + i, None, 1.0, 1.0, 1, (0,), 0)  # type: ignore[arg-type]
        for i in range(n)
    ]


def test_classify_keeps_eligible_flags_only_unlisted_in_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = get_config()
    day = date.today() - timedelta(days=10)  # recent -> routine band (±30%, 90 DTE)
    expiry = day + timedelta(days=30)
    atm = _osi(expiry, 100.0)          # ATM, standard, IN the listing -> kept
    unlisted = _osi(expiry, 105.0)     # in-band but NOT in the listing -> ENUMERATION_MISS
    far = _osi(expiry, 300.0)          # out of band, but listed -> dropped, not a miss

    listing = [_cref(atm, expiry, 100.0), _cref(far, expiry, 300.0)]
    monkeypatch.setattr(archive, "fetch_massive_contracts", lambda *a, **k: listing)

    by_symbol = {atm: _prints(atm, 2), unlisted: _prints(unlisted, 3), far: _prints(far, 1)}
    ch = _FakeCH([])
    keep, misses = archive._classify_day(
        by_symbol, {"NVDA": 100.0}, cfg, "k", ch, day, {}
    )

    assert set(keep) == {atm}          # only the in-band, listed contract is kept
    assert misses == 3                 # the unlisted in-band contract's prints
    # the full listing was written to option_contract_asof (self-describing reference)
    assert any(name == "option_contract_asof" for (name, _d, _c, _db) in ch.inserts)


def test_no_spot_underlying_is_skipped_not_missed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = get_config()
    day = date.today() - timedelta(days=10)
    expiry = day + timedelta(days=30)
    atm = _osi(expiry, 100.0)
    monkeypatch.setattr(archive, "fetch_massive_contracts", lambda *a, **k: [])
    keep, misses = archive._classify_day({atm: _prints(atm, 2)}, {}, cfg, "k", _FakeCH([]), day, {})
    assert keep == {} and misses == 0  # no daily bar -> can't judge, skip (never a miss)


# -- download cleanup ---------------------------------------------------------


class _OkS3:
    def download_fileobj(self, Bucket: str, Key: str, Fileobj: Any, Config: Any = None) -> None:
        Fileobj.write(b"data")


class _FailS3:
    def download_fileobj(self, Bucket: str, Key: str, Fileobj: Any, Config: Any = None) -> None:
        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")


def test_download_cleans_tempfile_on_success() -> None:
    cfg = get_config()
    seen: dict[str, Path] = {}
    with archive._download_day(_OkS3(), cfg, date(2026, 1, 5)) as path:
        seen["p"] = path
        assert path.exists() and path.read_bytes() == b"data"
    assert not seen["p"].exists()  # unlinked on the way out


def test_download_exhausts_and_leaves_no_tempfile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(archive.time, "sleep", lambda *_a: None)  # no real backoff wait
    cfg = get_config()
    with pytest.raises(archive.DownloadExhausted):
        with archive._download_day(_FailS3(), cfg, date(2026, 1, 5)):
            pass  # never reached
    # every attempt unlinked its partial file — nothing left in the temp dir
    import tempfile
    leftovers = list(Path(tempfile.gettempdir()).glob("optarch_2026-01-05_*"))
    assert leftovers == []


# -- quote phase --------------------------------------------------------------


def test_quotes_skipped_before_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = get_config()  # quotes_available_from = 2022-03-07
    calls: list[int] = []
    monkeypatch.setattr(
        archive, "fetch_option_quotes_day",
        lambda *a, **k: calls.append(1) or QuotePull([], 0, 0),
    )
    osi = _osi(date(2020, 6, 30), 100.0)
    q, pages, fetched = archive._quotes_for_day(cfg, {osi: _prints(osi, 2)}, date(2020, 1, 15))
    # pre-2022: no vendor quotes exist, no fetch attempted, zero pull totals
    assert q == {} and calls == [] and pages == 0 and fetched == 0


def test_quotes_pulled_and_attached_after_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    from option_archive.ingest_day import NbboQuote
    cfg = get_config()
    osi = _osi(date(2022, 5, 20), 100.0)
    prints = _prints(osi, 1)
    monkeypatch.setattr(
        archive, "fetch_option_quotes_day",
        lambda api, o, d: QuotePull(
            [NbboQuote(prints[0].sip_timestamp_ns - 1000, 1.0, 1.1, 2, 3)], pages=3, quotes_fetched=7
        ),
    )
    q, pages, fetched = archive._quotes_for_day(cfg, {osi: prints}, date(2022, 3, 8))
    ((_key, v),) = q.items()
    assert v.bid == 1.0 and v.ask == 1.1  # last quote at-or-before the print, attached
    assert pages == 3 and fetched == 7  # pull totals summed across contracts (one here)
