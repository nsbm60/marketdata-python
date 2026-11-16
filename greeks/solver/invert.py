"""IV inversion via Jäckel's native "Let's Be Rational" (greeks.solver._native), reason-coded.

``invert_batch`` is the one solver path: the pre-checks (t_below_floor, missing forward inputs,
below_intrinsic, above_no_arb, quantized_near_intrinsic) are applied as array masks with exactly the
thresholds and order the scalar code used, then one native call inverts the rows still live. The
scalar ``invert`` is a thin n=1 wrapper (IV from the batch, then greeks_batch), so live and history
share one path. The native library loads at import (greeks.solver._native) — no pure-Python fallback.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from greeks.domain import (
    FailureReason,
    OptionRight,
    RowStatus,
    SolverInput,
    SolverResult,
)
from greeks.solver._native import implied_vol_batch
from greeks.solver.black76 import greeks_batch

# Default OPRA-style tick for equity options under $3 is often 0.01; used only
# for the quantized-near-intrinsic short-T gate (not a silent price default).
DEFAULT_TICK_SIZE = 0.01
_ABS_EPS = 1e-10


def invert_batch(
    trade_price: npt.NDArray[np.float64],
    strike: npt.NDArray[np.float64],
    is_call: npt.NDArray[np.bool_],
    forward: npt.NDArray[np.float64],
    discount: npt.NDArray[np.float64],
    time_to_expiry: npt.NDArray[np.float64],
    *,
    methodology_version: str,
    t_floor_minutes: float = 15.0,
    tick_size: float = DEFAULT_TICK_SIZE,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.object_], npt.NDArray[np.object_]]:
    """Invert a batch of trades to IV, with per-row status and reason. Never a default IV on failure.

    Returns ``(iv, status, reason)``: ``iv`` is NaN on any failure; ``status`` is RowStatus.SUCCESS /
    FAILURE; ``reason`` is the FailureReason on failure, None on success. The masks below are the exact
    checks and order of the former scalar code — a row takes the first reason that applies.
    """
    t = time_to_expiry
    n = int(trade_price.shape[0])
    iv = np.full(n, np.nan, dtype=np.float64)
    reason: npt.NDArray[np.object_] = np.full(n, None, dtype=object)
    live = np.ones(n, dtype=bool)
    t_floor_years = (t_floor_minutes * 60.0) / (365.0 * 86400.0)

    def retire(mask: npt.NDArray[np.bool_], rc: FailureReason) -> None:
        m = live & mask
        reason[m] = rc
        live[m] = False

    with np.errstate(all="ignore"):  # already-retired rows may carry inf/nan inputs; masks ignore them
        retire(t < t_floor_years, FailureReason.T_BELOW_FLOOR)
        retire(~np.isfinite(trade_price) | (trade_price < 0), FailureReason.SOLVER_FAILED)
        retire(~np.isfinite(forward) | (forward <= 0), FailureReason.MISSING_FORWARD_INPUTS)
        retire(~np.isfinite(strike) | (strike <= 0), FailureReason.MISSING_FORWARD_INPUTS)
        retire(~np.isfinite(discount) | (discount <= 0), FailureReason.MISSING_FORWARD_INPUTS)

        intrinsic = np.where(is_call, np.maximum(forward - strike, 0.0),
                             np.maximum(strike - forward, 0.0)) * discount
        upper = np.where(is_call, forward, strike) * discount
        retire(trade_price < intrinsic - _ABS_EPS, FailureReason.BELOW_INTRINSIC)
        retire(trade_price > upper + _ABS_EPS, FailureReason.ABOVE_NO_ARB)
        one_day = 1.0 / 365.0
        retire((t < one_day) & (trade_price <= intrinsic + tick_size + _ABS_EPS),
               FailureReason.QUANTIZED_NEAR_INTRINSIC)

        idx = np.where(live)[0]
        if idx.size:
            d, tt, p = discount[idx], t[idx], trade_price[idx]
            # Undiscount exactly as py_vollib did: deflater = exp(-r t), r = -ln(D)/t, und = price/deflater.
            r = -np.log(d) / tt
            und = p / np.exp(-r * tt)
            iv[idx] = implied_vol_batch(und, forward[idx], strike[idx], tt,
                                        is_call[idx].astype(np.int32))
        retire(~np.isfinite(iv) | (iv <= 0), FailureReason.SOLVER_FAILED)

    # Build status as an object array by hand: RowStatus is a str subclass, so np.where would coerce
    # it to a unicode array and break identity checks against the enum.
    status: npt.NDArray[np.object_] = np.empty(n, dtype=object)
    status[live] = RowStatus.SUCCESS
    status[~live] = RowStatus.FAILURE
    return iv, status, reason


def invert(
    inp: SolverInput,
    *,
    methodology_version: str,
    t_floor_minutes: float = 15.0,
    tick_size: float = DEFAULT_TICK_SIZE,
) -> SolverResult:
    """Invert a single trade to IV + greeks, or a failure with reason_code — an n=1 call to the batch
    core. Never returns a default IV on failure."""
    is_call = np.array([inp.right is OptionRight.CALL], dtype=bool)
    iv, status, reason = invert_batch(
        np.array([inp.trade_price], dtype=np.float64), np.array([inp.strike], dtype=np.float64),
        is_call, np.array([inp.forward], dtype=np.float64),
        np.array([inp.discount], dtype=np.float64), np.array([inp.time_to_expiry], dtype=np.float64),
        methodology_version=methodology_version, t_floor_minutes=t_floor_minutes, tick_size=tick_size)
    if status[0] is RowStatus.FAILURE:
        return _fail(methodology_version, reason[0])

    iv0 = float(iv[0])
    g = greeks_batch(
        np.array([inp.forward], dtype=np.float64), np.array([inp.strike], dtype=np.float64),
        np.array([inp.time_to_expiry], dtype=np.float64), np.array([inp.discount], dtype=np.float64),
        np.array([iv0], dtype=np.float64), is_call,
        np.array([np.nan if inp.spot is None else inp.spot], dtype=np.float64))
    return SolverResult(
        status=RowStatus.SUCCESS, methodology_version=methodology_version, iv=iv0,
        delta=float(g.delta[0]), gamma=float(g.gamma[0]), vega=float(g.vega[0]),
        theta=float(g.theta[0]), rho=float(g.rho[0]), reason_code=None,
        theta_interpretively_limited=bool(g.theta_interpretively_limited[0]))


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
