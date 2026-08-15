"""PR4: contract filters, staging, alpaca RAW guard (offline)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from alpaca.data.enums import Adjustment

from greeks.domain import FailureReason, OptionRight
from greeks.pull.alpaca_spot import (
    REQUIRED_ADJUSTMENT,
    EquityTradePrint,
    assert_raw_adjustment,
    attach_spot_to_option_trades,
    spot_asof,
)
from greeks.pull.contracts import (
    ContractRef,
    contracts_from_clickhouse_rows,
    filter_contracts,
    is_in_moneyness_band,
)
from greeks.pull.massive_trades import OptionTradePrint, ns_to_utc
from greeks.pull.staging import TradeStaging


def _ref(
    osi: str,
    *,
    strike: float,
    expiry: date,
    shares: int = 100,
    right: OptionRight = OptionRight.CALL,
) -> ContractRef:
    return ContractRef(
        osi=osi,
        massive_ticker=f"O:{osi}",
        underlying="NVDA",
        expiry=expiry,
        strike=strike,
        right=right,
        shares_per_contract=shares,
        exercise_style="american",
        primary_exchange=None,
    )


def test_moneyness_band() -> None:
    assert is_in_moneyness_band(100.0, 100.0, 0.30)
    assert is_in_moneyness_band(130.0, 100.0, 0.30)
    assert not is_in_moneyness_band(131.0, 100.0, 0.30)
    assert not is_in_moneyness_band(69.0, 100.0, 0.30)


def test_filter_dte_moneyness_deliverable() -> None:
    as_of = date(2026, 5, 27)
    spot = 100.0
    contracts = [
        _ref("NVDA260626C00100000", strike=100.0, expiry=date(2026, 6, 26)),  # ok
        _ref("NVDA260626C00200000", strike=200.0, expiry=date(2026, 6, 26)),  # mny
        _ref("NVDA261218C00100000", strike=100.0, expiry=date(2026, 12, 18)),  # dte
        _ref(
            "NVDA260626C00100000A",
            strike=100.0,
            expiry=date(2026, 6, 26),
            shares=1000,
        ),  # nonstandard — bad OSI actually; use valid osi
    ]
    # Fix nonstandard with valid OSI-like name
    contracts[3] = _ref(
        "NVDA260626C00105000",
        strike=105.0,
        expiry=date(2026, 6, 26),
        shares=1000,
    )
    res = filter_contracts(contracts, as_of=as_of, spot=spot)
    assert len(res.eligible) == 1
    assert res.eligible[0].strike == 100.0
    assert res.skipped_moneyness == 1
    assert res.skipped_dte == 1
    assert len(res.nonstandard) == 1
    assert res.nonstandard[0][1] is FailureReason.NONSTANDARD_CONTRACT


def test_adjustment_raw_constant() -> None:
    assert REQUIRED_ADJUSTMENT is Adjustment.RAW
    assert assert_raw_adjustment() is Adjustment.RAW


def test_spot_asof_and_attach() -> None:
    eq = [
        EquityTradePrint(
            "NVDA",
            datetime(2026, 5, 27, 14, 0, tzinfo=timezone.utc),
            100.0,
            10,
        ),
        EquityTradePrint(
            "NVDA",
            datetime(2026, 5, 27, 15, 0, tzinfo=timezone.utc),
            101.0,
            10,
        ),
    ]
    assert spot_asof(eq, datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)).price == 100.0  # type: ignore[union-attr]
    assert spot_asof(eq, datetime(2026, 5, 27, 16, 0, tzinfo=timezone.utc)).price == 101.0  # type: ignore[union-attr]
    assert spot_asof(eq, datetime(2026, 5, 27, 13, 0, tzinfo=timezone.utc)) is None

    ot = OptionTradePrint(
        symbol="NVDA260527C00100000",
        trade_ts=datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc),
        price=5.0,
        size=1,
        exchange=1,
        conditions=(),
        sequence_number=1,
        sip_timestamp_ns=1_748_000_000_000_000_000,
    )
    pairs = attach_spot_to_option_trades([ot], eq, "NVDA")
    assert len(pairs) == 1
    assert pairs[0][1] is not None
    assert pairs[0][1].spot == 100.0


def test_staging_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "stage.db"
    ot = OptionTradePrint(
        symbol="NVDA260527C00100000",
        trade_ts=datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc),
        price=5.25,
        size=2,
        exchange=46,
        conditions=(209,),
        sequence_number=7,
        sip_timestamp_ns=1_748_000_000_000_000_001,
    )
    from greeks.pull.alpaca_spot import SpotAtTrade

    spot = SpotAtTrade(
        option_symbol=ot.symbol,
        option_trade_ts=ot.trade_ts,
        spot=135.5,
        spot_trade_ts=datetime(2026, 5, 27, 14, 29, tzinfo=timezone.utc),
        underlying="NVDA",
    )
    with TradeStaging(db) as st:
        n = st.upsert_trades(date(2026, 5, 27), "NVDA", [(ot, spot)])
        assert n == 1
        assert st.count(underlying="NVDA", session_date=date(2026, 5, 27)) == 1
        rows = list(st.iter_session("NVDA", date(2026, 5, 27)))
        assert len(rows) == 1
        assert rows[0].price == 5.25
        assert rows[0].spot_at_trade == 135.5


def test_ns_to_utc() -> None:
    ts = ns_to_utc(1_400_000_000_000_000_000)
    assert ts.tzinfo is not None
    assert ts.year >= 2014


def test_ch_rows_mapper() -> None:
    rows = [
        {
            "option_symbol": "NVDA260626C00100000",
            "expiration_date": date(2026, 6, 26),
            "strike_price": 100.0,
            "call_put": "C",
            "contract_size": 100,
        }
    ]
    refs = contracts_from_clickhouse_rows(rows, "NVDA")
    assert len(refs) == 1
    assert refs[0].osi == "NVDA260626C00100000"
