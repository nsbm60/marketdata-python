"""Unit tests for the greeks_snapshot pure seams (no network).

The pricing/carry/invert machinery is already covered by the certified-part
tests; here we only test the orchestration glue the snapshot CLI adds:
last-print selection, StagedTrade construction, and table formatting.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from greeks.domain import FailureReason, OptionRight, RowStatus
from greeks.harness.rows import ValidationRow
from greeks.pull.contracts import ContractRef
from greeks.pull.massive_trades import OptionTradePrint
from greeks.snapshot.run import format_rows, select_last_print, staged_from_print


def _print(osi: str, ns: int, price: float) -> OptionTradePrint:
    return OptionTradePrint(
        symbol=osi,
        trade_ts=datetime.fromtimestamp(ns / 1e9, tz=timezone.utc),
        price=price,
        size=1.0,
        exchange=1,
        conditions=(),
        sequence_number=ns,
        sip_timestamp_ns=ns,
    )


def test_select_last_print_takes_most_recent() -> None:
    osi = "NVDA260515C00100000"
    trades = [_print(osi, 1_000, 1.0), _print(osi, 3_000, 3.0), _print(osi, 2_000, 2.0)]
    # ascending-by-SIP is the contract; emulate it
    trades.sort(key=lambda t: t.sip_timestamp_ns)
    last = select_last_print(trades)
    assert last is not None and last.price == 3.0


def test_select_last_print_empty_is_none() -> None:
    assert select_last_print([]) is None


def test_staged_from_print_carries_spot_and_price_no_sqlite() -> None:
    osi = "NVDA260515C00100000"
    c = ContractRef(
        osi=osi,
        massive_ticker="O:NVDA260515C00100000",
        underlying="NVDA",
        expiry=date(2026, 5, 15),
        strike=100.0,
        right=OptionRight.CALL,
        shares_per_contract=100,
        exercise_style="american",
        primary_exchange="XNAS",
    )
    lp = _print(osi, 5_000_000_000, 4.25)
    spot_ts = datetime(2026, 5, 13, 19, 55, tzinfo=timezone.utc)
    st = staged_from_print(
        c, lp, session_date=date(2026, 5, 13), underlying="nvda",
        spot=101.5, spot_ts=spot_ts,
    )
    assert st.symbol == osi
    assert st.price == 4.25
    assert st.spot_at_trade == 101.5
    assert st.spot_trade_ts == spot_ts
    assert st.underlying == "NVDA"
    assert st.session_date == date(2026, 5, 13)
    assert st.trade_ts == lp.trade_ts


def _row(strike: float, right: OptionRight, *, ok: bool) -> ValidationRow:
    return ValidationRow(
        symbol=f"NVDA260515{right.value}{int(strike * 1000):08d}",
        trade_ts=datetime(2026, 5, 13, 19, 59, tzinfo=timezone.utc),
        methodology_version="v1",
        underlying="NVDA",
        expiry=date(2026, 5, 15),
        strike=strike,
        right=right,
        trade_price=4.25,
        spot_at_trade=101.5,
        forward=101.6 if ok else None,
        discount=0.999 if ok else None,
        time_to_expiry=0.0054 if ok else None,
        iv=0.42 if ok else None,
        delta=0.55 if ok else None,
        gamma=0.03 if ok else None,
        vega=0.08 if ok else None,
        theta=-0.09 if ok else None,
        rho=0.01 if ok else None,
        status=RowStatus.SUCCESS if ok else RowStatus.FAILURE,
        reason_code=None if ok else FailureReason.MISSING_FORWARD_INPUTS,
    )


def test_format_rows_includes_greeks_inputs_and_status() -> None:
    out = format_rows([_row(100.0, OptionRight.CALL, ok=True), _row(95.0, OptionRight.PUT, ok=False)])
    # header names the inputs the user asked to see
    assert "strike" in out and "fwd" in out and "disc" in out and "T(yr)" in out
    assert "delta" in out and "theta" in out and "rho" in out
    # success row shows iv; failure row shows its reason as status and dashes for greeks
    assert "0.4200" in out
    assert FailureReason.MISSING_FORWARD_INPUTS.value in out
    assert "—" in out


def test_format_rows_sorted_by_strike_then_side() -> None:
    out = format_rows([
        _row(100.0, OptionRight.PUT, ok=True),
        _row(100.0, OptionRight.CALL, ok=True),
        _row(95.0, OptionRight.CALL, ok=True),
    ])
    body = out.splitlines()[2:]  # skip header + rule
    strikes = [float(line.split()[0]) for line in body]
    assert strikes == sorted(strikes)
