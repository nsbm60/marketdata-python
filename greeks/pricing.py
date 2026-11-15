"""The shared option-pricing core (greeks-server-spec.md §2, §5.4, §7). Pure.

Given a price and a spot — whatever their basis: a two-sided quote mid for the live server, a trade
print for the historical job — invert IV and the greeks under Black-76: carry (forward/discount/TTE)
from the SOFR rate and dividends, then greeks.solver.invert. Live (greeks_server.compute) and
historical (greeks.history) share THIS ONE core; there is no second inversion path. Only the price
basis differs, and the caller records it. Nothing is defaulted — a failure carries the solver's
reason with the greeks left absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, Optional, Sequence

from greeks.domain import Dividend, FailureReason, OptionRight, RowStatus, SolverInput
from greeks.forwards.carry import carry
from greeks.solver.invert import invert


@dataclass(frozen=True)
class PricedGreeks:
    """The core's result for one option: a success with IV + greeks, or a failure reason with the
    greeks absent. ``forward``/``discount``/``tte`` are present once the carry was computed (so they
    are set on a solver-stage failure, absent on an expiry-before-session-date failure)."""
    status: RowStatus
    reason: Optional[FailureReason]
    iv: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    rho: Optional[float]
    forward: Optional[float]
    discount: Optional[float]
    tte: Optional[float]
    theta_limited: bool = False


def _fail(reason: FailureReason, forward: Optional[float], discount: Optional[float],
          tte: Optional[float]) -> PricedGreeks:
    return PricedGreeks(RowStatus.FAILURE, reason, None, None, None, None, None, None, forward, discount, tte)


def price_option(
    *,
    expiry: date,
    right: OptionRight,
    strike: float,
    price: float,
    spot: float,
    as_of: datetime,
    session_date: date,
    rate_for: Callable[[date], float],
    dividends: Sequence[Dividend],
    methodology_version: str,
    t_floor_minutes: float,
    expiry_time_et: str,
) -> PricedGreeks:
    """Invert IV + greeks for one option at (``price``, ``spot``) as of ``as_of``. ``rate_for`` and
    ``dividends`` are for ``session_date``. An expiry before the session date, or any solver reason
    (below_intrinsic, above_no_arb, t_below_floor, …), is a failure with the greeks absent."""
    if expiry < session_date:
        # Expired before the session date: no rate for a negative term; the solver's floor reason is
        # what it would give. No carry, so no forward/discount/tte.
        return _fail(FailureReason.T_BELOW_FLOOR, None, None, None)
    c = carry(spot, rate_for(expiry), as_of, expiry, dividends, expiry_time_et=expiry_time_et)
    res = invert(
        SolverInput(trade_price=price, strike=strike, right=right, forward=c.forward,
                    discount=c.discount, time_to_expiry=c.time_to_expiry, spot=spot),
        methodology_version=methodology_version, t_floor_minutes=t_floor_minutes)
    if res.status is not RowStatus.SUCCESS:
        assert res.reason_code is not None  # SolverResult invariant
        return _fail(res.reason_code, c.forward, c.discount, c.time_to_expiry)
    return PricedGreeks(
        status=RowStatus.SUCCESS, reason=None,
        iv=res.iv, delta=res.delta, gamma=res.gamma, theta=res.theta, vega=res.vega, rho=res.rho,
        forward=c.forward, discount=c.discount, tte=c.time_to_expiry,
        theta_limited=res.theta_interpretively_limited)
