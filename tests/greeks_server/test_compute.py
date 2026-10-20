"""compute(): the solver is the only pricing authority; failures carry what is known."""

from __future__ import annotations

from datetime import date

import pytest

from greeks.domain import Dividend, FailureReason, RowStatus, SolverInput
from greeks.forwards.carry import carry
from greeks.solver.black76 import price as black_price
from greeks.solver.invert import invert
from greeks_server.compute import Contract, compute
from greeks_server.price import PriceKind, SelectedPrice

from gs_helpers import M, NOW, NOW_MS, SESSION, inputs

C = Contract.parse("NVDA261120C00180000")
SPOT = SelectedPrice(181.0, PriceKind.MID, NOW_MS - 100)


def _fair(c: Contract, vol: float, divs: tuple[Dividend, ...] = ()) -> float:
    inp = inputs({c.underlying: divs} if divs else None)
    cr = carry(SPOT.value, inp.rate_for(c.expiry), NOW, c.expiry, divs)
    return black_price(cr.forward, c.strike, cr.time_to_expiry, cr.discount, vol, c.right)


def test_contract_parse() -> None:
    assert (C.underlying, C.expiry, C.side, C.strike) == ("NVDA", date(2026, 11, 20), "C", 180.0)
    with pytest.raises(ValueError):
        Contract.parse("NVDA1261120C00180000")


def test_success_matches_the_solver_directly() -> None:
    px = SelectedPrice(_fair(C, 0.45), PriceKind.MID, NOW_MS - 200)
    inp = inputs()
    msg = compute(C, px, SPOT, inp, NOW, M)
    assert msg.status is RowStatus.SUCCESS
    assert msg.iv == pytest.approx(0.45, rel=1e-9)
    cr = carry(SPOT.value, inp.rate_for(C.expiry), NOW, C.expiry, ())
    direct = invert(
        SolverInput(px.value, C.strike, C.right, cr.forward, cr.discount, cr.time_to_expiry, SPOT.value),
        methodology_version="v2",
    )
    assert (msg.iv, msg.delta, msg.gamma, msg.theta, msg.vega, msg.rho) == (
        direct.iv, direct.delta, direct.gamma, direct.theta, direct.vega, direct.rho,
    )
    assert (msg.forward, msg.discount, msg.t_years) == (cr.forward, cr.discount, cr.time_to_expiry)
    assert (msg.price, msg.price_kind, msg.price_ts) == (px.value, PriceKind.MID, px.ts_ms)
    assert msg.computed_at == NOW_MS and msg.methodology_version == "v2"


def test_dividend_inside_life_lowers_the_forward() -> None:
    div = (Dividend("NVDA", 0.5, date(2026, 11, 5)),)
    px = SelectedPrice(_fair(C, 0.45, div), PriceKind.MID, NOW_MS)
    with_div = compute(C, px, SPOT, inputs({"NVDA": div}), NOW, M)
    without = compute(C, px, SPOT, inputs(), NOW, M)
    assert with_div.forward is not None and without.forward is not None
    assert with_div.forward < without.forward
    assert with_div.iv == pytest.approx(0.45, rel=1e-9)


def test_missing_spot_first() -> None:
    px = SelectedPrice(9.0, PriceKind.LAST, NOW_MS)
    msg = compute(C, px, None, inputs(), NOW, M)
    assert msg.reason is FailureReason.MISSING_SPOT
    assert msg.price == 9.0 and msg.spot is None and msg.forward is None


def test_no_session_price() -> None:
    msg = compute(C, None, SPOT, inputs(), NOW, M)
    assert msg.reason is FailureReason.NO_SESSION_PRICE
    assert msg.price is None and msg.spot == SPOT.value
    assert "price" not in msg.to_wire() and "iv" not in msg.to_wire()


def test_solver_failure_carries_carry_inputs() -> None:
    deep = Contract.parse("NVDA261120C00100000")
    msg = compute(deep, SelectedPrice(1.0, PriceKind.LAST, NOW_MS), SPOT, inputs(), NOW, M)
    assert msg.reason is FailureReason.BELOW_INTRINSIC
    assert msg.forward is not None and msg.discount is not None and msg.t_years is not None


def test_last_fifteen_minutes_is_the_solver_floor() -> None:
    zero_dte = Contract.parse("NVDA261020C00180000")
    late = NOW.replace(hour=19, minute=50)  # 15:50 ET, 10 min to expiry
    msg = compute(zero_dte, SelectedPrice(1.2, PriceKind.MID, NOW_MS), SPOT, inputs(), late, M)
    assert msg.reason is FailureReason.T_BELOW_FLOOR


def test_expired_contract() -> None:
    old = Contract.parse("NVDA261016C00180000")
    msg = compute(old, SelectedPrice(1.2, PriceKind.MID, NOW_MS), SPOT, inputs(), NOW, M)
    assert msg.reason is FailureReason.T_BELOW_FLOOR
    assert SESSION > old.expiry
