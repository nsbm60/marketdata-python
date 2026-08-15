"""Tier-2 independent Black-76 via QuantLib (optional dependency).

Uses QuantLib's closed-form ``blackFormula`` / ``BlackCalculator`` — not our
py_vollib path — so year-fraction, discounting placement, and sign conventions
can be cross-checked.

Requires: ``pip install '.[tier2]'`` or ``pip install QuantLib``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from greeks.domain import OptionRight
from greeks.solver.black76 import rate_from_discount

if TYPE_CHECKING:
    pass

# Lazy import so machines without QuantLib still import this package under mypy
# when tests skip. Runtime calls raise ImportError with install hint.


def _ql() -> Any:
    try:
        import QuantLib as ql
    except ImportError as e:
        raise ImportError(
            "QuantLib is required for Tier-2. Install with: "
            "pip install '.[tier2]'  or  pip install QuantLib"
        ) from e
    return ql


def _option_type(right: OptionRight, ql: Any) -> Any:
    return ql.Option.Call if right is OptionRight.CALL else ql.Option.Put


def _payoff(right: OptionRight, strike: float, ql: Any) -> Any:
    return ql.PlainVanillaPayoff(_option_type(right, ql), strike)


@dataclass(frozen=True)
class QuantLibBlackGreeks:
    """Greeks aligned to methodology units (see tier2_conventions.md)."""

    price: float
    delta: float  # forward delta; equals spot delta when S=F
    gamma: float
    vega: float  # per 1 vol point (1%), same as py_vollib
    theta: float  # per calendar day, same as py_vollib / QL thetaPerDay
    rho_cash: float  # QuantLib BlackCalculator.rho / 100 (per 1% rate)
    rho_futures: float  # F-fixed futures rho / 100: -T * price / 100
    delta_is_spot: bool
    theta_interpretively_limited: bool


def price(
    forward: float,
    strike: float,
    time_to_expiry: float,
    discount: float,
    sigma: float,
    right: OptionRight,
) -> float:
    """Discounted Black-76 price via QuantLib ``blackFormula``."""
    if time_to_expiry <= 0:
        if right is OptionRight.CALL:
            return max(forward - strike, 0.0) * discount
        return max(strike - forward, 0.0) * discount
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    ql = _ql()
    std_dev = sigma * math.sqrt(time_to_expiry)
    return float(
        ql.blackFormula(
            _option_type(right, ql),
            strike,
            forward,
            std_dev,
            discount,
        )
    )


def greeks(
    forward: float,
    strike: float,
    time_to_expiry: float,
    discount: float,
    sigma: float,
    right: OptionRight,
    *,
    spot: Optional[float] = None,
) -> QuantLibBlackGreeks:
    """Analytic Black-76 greeks via QuantLib ``BlackCalculator``.

    Unit alignment to py_vollib methodology:
    - vega: QuantLib absolute vega / 100 (per 1 vol point)
    - theta: ``thetaPerDay`` (per calendar day)
    - rho_cash: QuantLib ``rho(T)`` / 100 — BS cash rho (T D K N(d2) style);
      **not** the same quantity as py_vollib black rho
    - rho_futures: F held fixed, ``-T * price / 100`` — matches py_vollib black rho
    - delta: forward delta; if ``spot`` given, convert via F/S to spot terms
    """
    if time_to_expiry <= 0:
        if right is OptionRight.CALL:
            px = max(forward - strike, 0.0) * discount
        else:
            px = max(strike - forward, 0.0) * discount
        return QuantLibBlackGreeks(
            price=px,
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            rho_cash=0.0,
            rho_futures=0.0,
            delta_is_spot=spot is not None,
            theta_interpretively_limited=True,
        )

    ql = _ql()
    std_dev = sigma * math.sqrt(time_to_expiry)
    bc = ql.BlackCalculator(_payoff(right, strike, ql), forward, std_dev, discount)
    px = float(bc.value())
    d_fwd = float(bc.deltaForward())
    if spot is not None and spot > 0:
        d = d_fwd * (forward / spot)
        delta_is_spot = True
    else:
        d = d_fwd
        delta_is_spot = False

    limited = time_to_expiry * 365.0 < 1.0
    return QuantLibBlackGreeks(
        price=px,
        delta=d,
        gamma=float(bc.gamma(forward)),
        vega=float(bc.vega(time_to_expiry)) / 100.0,
        theta=float(bc.thetaPerDay(forward, time_to_expiry)),
        rho_cash=float(bc.rho(time_to_expiry)) / 100.0,
        rho_futures=(-time_to_expiry * px) / 100.0,
        delta_is_spot=delta_is_spot,
        theta_interpretively_limited=limited,
    )


def implied_vol(
    trade_price: float,
    forward: float,
    strike: float,
    time_to_expiry: float,
    discount: float,
    right: OptionRight,
    *,
    vol_guess: float = 0.25,
    accuracy: float = 1e-14,
    max_iterations: int = 200,
) -> float:
    """Black-76 IV via QuantLib ``blackFormulaImpliedStdDev``.

    ``vol_guess`` seeds stdDev = vol_guess * sqrt(T). Short-dated points need a
    non-default guess for the solver to reach 1e-8 relative accuracy.
    """
    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be > 0 for implied_vol")
    if trade_price < 0:
        raise ValueError("trade_price must be >= 0")
    ql = _ql()
    sqrt_t = math.sqrt(time_to_expiry)
    guess_std = max(vol_guess, 1e-6) * sqrt_t
    std_dev = float(
        ql.blackFormulaImpliedStdDev(
            _option_type(right, ql),
            strike,
            forward,
            trade_price,
            discount,
            0.0,  # displacement
            guess_std,
            accuracy,
            max_iterations,
        )
    )
    return std_dev / sqrt_t


def rate_from_discount_ql(discount: float, time_to_expiry: float) -> float:
    """Same continuous-rate definition as ``black76.rate_from_discount``."""
    return rate_from_discount(discount, time_to_expiry)
