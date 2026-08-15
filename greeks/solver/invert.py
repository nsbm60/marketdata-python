"""Scalar IV inversion via py_vollib (Jäckel) with reason-coded failures."""

from __future__ import annotations

import math
from typing import Optional

from py_vollib.black.implied_volatility import (
    implied_volatility as _implied_volatility,
)

from greeks.domain import (
    FailureReason,
    OptionRight,
    RowStatus,
    SolverInput,
    SolverResult,
)
from greeks.solver.black76 import (
    forward_intrinsic,
    greeks as black_greeks,
    rate_from_discount,
    upper_no_arb_bound,
)

# Default OPRA-style tick for equity options under $3 is often 0.01; used only
# for the quantized-near-intrinsic short-T gate (not a silent price default).
DEFAULT_TICK_SIZE = 0.01
_REL_EPS = 1e-12
_ABS_EPS = 1e-10


def _flag(right: OptionRight) -> str:
    return "c" if right is OptionRight.CALL else "p"


def invert(
    inp: SolverInput,
    *,
    methodology_version: str,
    t_floor_minutes: float = 15.0,
    tick_size: float = DEFAULT_TICK_SIZE,
) -> SolverResult:
    """Invert a single trade to IV + greeks, or a failure with reason_code.

    Never returns a default IV on failure.
    """
    t = inp.time_to_expiry
    t_floor_years = (t_floor_minutes * 60.0) / (365.0 * 86400.0)

    if t < t_floor_years:
        return _fail(methodology_version, FailureReason.T_BELOW_FLOOR)

    if not math.isfinite(inp.trade_price) or inp.trade_price < 0:
        return _fail(methodology_version, FailureReason.SOLVER_FAILED)
    if not math.isfinite(inp.forward) or inp.forward <= 0:
        return _fail(methodology_version, FailureReason.MISSING_FORWARD_INPUTS)
    if not math.isfinite(inp.strike) or inp.strike <= 0:
        return _fail(methodology_version, FailureReason.MISSING_FORWARD_INPUTS)
    if not math.isfinite(inp.discount) or inp.discount <= 0:
        return _fail(methodology_version, FailureReason.MISSING_FORWARD_INPUTS)

    intrinsic = forward_intrinsic(inp.forward, inp.strike, inp.discount, inp.right)
    upper = upper_no_arb_bound(inp.forward, inp.strike, inp.discount, inp.right)
    price = inp.trade_price

    if price < intrinsic - _ABS_EPS:
        return _fail(methodology_version, FailureReason.BELOW_INTRINSIC)

    if price > upper + _ABS_EPS:
        return _fail(methodology_version, FailureReason.ABOVE_NO_ARB)

    # Ill-defined IV under quantization for very short-dated near-intrinsic prints.
    one_day = 1.0 / 365.0
    if t < one_day and price <= intrinsic + tick_size + _ABS_EPS:
        return _fail(methodology_version, FailureReason.QUANTIZED_NEAR_INTRINSIC)

    try:
        r = rate_from_discount(inp.discount, t)
        iv = float(
            _implied_volatility(
                price,
                inp.forward,
                inp.strike,
                r,
                t,
                _flag(inp.right),
            )
        )
    except Exception:
        return _fail(methodology_version, FailureReason.SOLVER_FAILED)

    if not math.isfinite(iv) or iv <= 0:
        return _fail(methodology_version, FailureReason.SOLVER_FAILED)

    g = black_greeks(
        inp.forward,
        inp.strike,
        t,
        inp.discount,
        iv,
        inp.right,
        spot=inp.spot,
    )

    return SolverResult(
        status=RowStatus.SUCCESS,
        methodology_version=methodology_version,
        iv=iv,
        delta=g.delta,
        gamma=g.gamma,
        vega=g.vega,
        theta=g.theta,
        rho=g.rho,
        reason_code=None,
        theta_interpretively_limited=g.theta_interpretively_limited,
    )


def _fail(methodology_version: str, reason: FailureReason) -> SolverResult:
    return SolverResult(
        status=RowStatus.FAILURE,
        methodology_version=methodology_version,
        reason_code=reason,
    )


def relative_iv_error(recovered: float, true_vol: float) -> float:
    """Relative |σ̂ − σ| / σ. ``true_vol`` must be > 0."""
    if true_vol <= 0:
        raise ValueError("true_vol must be > 0")
    return abs(recovered - true_vol) / true_vol
