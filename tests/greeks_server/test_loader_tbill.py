"""clickhouse_loader: the pre-SOFR overnight series comes from the 1-month T-bill (ACT/360), a
post-SOFR gap still fails, and a missing pre-SOFR bill fails naming the date. A fake ClickHouse
client routes queries by SQL; no live database."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pytest

from greeks_server.inputs import clickhouse_loader

FIRST_SOFR = date(2018, 4, 2)


class _Result:
    def __init__(self, rows: list[tuple[Any, ...]], cols: list[str] | None = None) -> None:
        self.result_rows = rows
        self.column_names = cols or []


class FakeCH:
    """Answers the four reads clickhouse_loader makes, keyed off the SQL text."""

    def __init__(
        self,
        *,
        first_sofr: date | None,
        sofr_rows: list[tuple[date, float]],
        tbill_rows: list[tuple[date, float]],
    ) -> None:
        self._first_sofr = first_sofr
        self._sofr = sofr_rows
        self._tbill = tbill_rows

    def query(self, sql: str, parameters: dict[str, Any] | None = None) -> _Result:
        p = parameters or {}
        if "min(date)" in sql:  # first_sofr_date_from_ch
            return _Result([(self._first_sofr,)])
        if "treasury_yields" in sql:  # load_tbill_1m_series_from_ch
            lo, hi = p["start"], p["end"]
            return _Result([(d, y) for d, y in self._tbill if lo <= d <= hi])
        if "sofr_daily" in sql:  # load_sofr_series_from_ch (the series read)
            lo, hi = p.get("start"), p.get("end")
            rows = [(d, r, "nyfed") for d, r in self._sofr
                    if (lo is None or d >= lo) and (hi is None or d <= hi)]
            return _Result(rows, cols=["date", "rate", "source"])
        if "trading.dividend" in sql:  # load_upcoming_dividends
            return _Result([])
        raise AssertionError(f"unexpected query: {sql}")


def test_pre_sofr_session_uses_the_tbill(caplog: pytest.LogCaptureFixture) -> None:
    session = date(2016, 1, 4)  # entirely before the first SOFR fix
    bills = [(session - timedelta(days=i), 0.0036) for i in range(0, 31)]
    ch = FakeCH(first_sofr=FIRST_SOFR, sofr_rows=[], tbill_rows=bills)
    with caplog.at_level(logging.INFO, logger="greeks_server.inputs"):
        si = clickhouse_loader(lambda: ch)(session)
    assert si.sofr[session] == pytest.approx(0.0036 * 360.0 / 365.0, rel=1e-12)  # converted bill
    assert all(d < FIRST_SOFR for d in si.sofr)  # no SOFR in a 2016 series
    assert "rate=TBILL1M" in caplog.text


def test_post_sofr_gap_still_fails() -> None:
    session = date(2026, 10, 20)  # modern: window is well after the first SOFR fix
    sofr = [(session - timedelta(days=11), 0.043)]  # newest fix 11 days stale — a gap
    ch = FakeCH(first_sofr=FIRST_SOFR, sofr_rows=sofr, tbill_rows=[])
    with pytest.raises(ValueError, match="more than 10 days"):
        clickhouse_loader(lambda: ch)(session)


def test_pre_sofr_missing_bill_fails_naming_the_date() -> None:
    session = date(2016, 1, 4)
    ch = FakeCH(first_sofr=FIRST_SOFR, sofr_rows=[], tbill_rows=[])  # no bill anywhere
    with pytest.raises(ValueError, match=str(session)):
        clickhouse_loader(lambda: ch)(session)
