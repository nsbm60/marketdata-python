"""Each invert failure reason has a constructed path."""

from __future__ import annotations

import math

from greeks.domain import FailureReason, OptionRight, RowStatus, SolverInput
from greeks.solver.black76 import price as black_price
from greeks.solver.invert import invert

_MV = "v1"
_F = 100.0
_K = 100.0
_D = math.exp(-0.05 * (30 / 365))
_T_30D = 30 / 365
_T_1H = 1 / 365 / 24


def _base(**overrides: object) -> SolverInput:
    data: dict[str, object] = dict(
        trade_price=2.5,
        strike=_K,
        right=OptionRight.CALL,
        forward=_F,
        discount=_D,
        time_to_expiry=_T_30D,
        spot=_F,
    )
    data.update(overrides)
    return SolverInput(**data)  # type: ignore[arg-type]


def test_t_below_floor() -> None:
    # 10 minutes < 15 minute floor
    t = (10 * 60) / (365 * 86400)
    r = invert(_base(time_to_expiry=t), methodology_version=_MV, t_floor_minutes=15)
    assert r.status is RowStatus.FAILURE
    assert r.reason_code is FailureReason.T_BELOW_FLOOR
    assert r.iv is None


def test_below_intrinsic() -> None:
    # Deep ITM call intrinsic ≈ (F-K)*D with F=110, K=100
    F, K = 110.0, 100.0
    D = _D
    intrinsic = (F - K) * D
    r = invert(
        _base(forward=F, strike=K, trade_price=intrinsic - 0.05, discount=D),
        methodology_version=_MV,
    )
    assert r.reason_code is FailureReason.BELOW_INTRINSIC


def test_above_no_arb() -> None:
    r = invert(
        _base(trade_price=_F * _D + 1.0),
        methodology_version=_MV,
    )
    assert r.reason_code is FailureReason.ABOVE_NO_ARB


def test_quantized_near_intrinsic_short_t() -> None:
    # ATM call short T: intrinsic 0; price at one tick
    r = invert(
        _base(
            trade_price=0.01,
            time_to_expiry=_T_1H,
            forward=100.0,
            strike=100.0,
        ),
        methodology_version=_MV,
        t_floor_minutes=0.0,  # allow short T past floor
        tick_size=0.01,
    )
    assert r.reason_code is FailureReason.QUANTIZED_NEAR_INTRINSIC


def test_success_roundtrip_path() -> None:
    sigma = 0.25
    px = black_price(_F, _K, _T_30D, _D, sigma, OptionRight.CALL)
    r = invert(_base(trade_price=px), methodology_version=_MV)
    assert r.status is RowStatus.SUCCESS
    assert r.iv is not None
    assert abs(r.iv - sigma) / sigma < 1e-10
