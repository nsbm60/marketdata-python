"""Black-76 forward-form pricing and analytic greeks.

Pricing and IV core use py_vollib (Jäckel). Array-shaped helpers take an ``xp``
module handle (default NumPy) for future GPU portability; scalar path is used
in tiers 1–2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from greeks.domain import OptionRight

# py_vollib re-exports vollib; keep import path per methodology brief.
from py_vollib.black import black as _black_price
from py_vollib.black.greeks.analytical import (
    delta as _delta,
    gamma as _gamma,
    rho as _rho,
    theta as _theta,
    vega as _vega,
)


def _flag(right: OptionRight) -> str:
    return "c" if right is OptionRight.CALL else "p"


def rate_from_discount(discount: float, time_to_expiry: float) -> float:
    """Continuous rate r such that D = exp(-r T)."""
    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be > 0 for rate_from_discount")
    if discount <= 0:
        raise ValueError("discount must be > 0")
    return -math.log(discount) / time_to_expiry


def forward_intrinsic(forward: float, strike: float, discount: float, right: OptionRight) -> float:
    """Discounted forward intrinsic value."""
    if right is OptionRight.CALL:
        return max(forward - strike, 0.0) * discount
    return max(strike - forward, 0.0) * discount


def upper_no_arb_bound(forward: float, strike: float, discount: float, right: OptionRight) -> float:
    """Simple Black-76 upper bound on discounted premium."""
    if right is OptionRight.CALL:
        return forward * discount
    return strike * discount


@dataclass(frozen=True)
class BlackGreeks:
    """Analytic greeks at a given vol. Delta is spot terms when spot is set."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float  # per calendar day (py_vollib convention)
    rho: float
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
    """Discounted Black-76 option price."""
    if time_to_expiry <= 0:
        return forward_intrinsic(forward, strike, discount, right)
    r = rate_from_discount(discount, time_to_expiry)
    return float(
        _black_price(_flag(right), forward, strike, time_to_expiry, r, sigma)
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
) -> BlackGreeks:
    """Analytic Black-76 greeks.

    - Delta: if ``spot`` is provided and > 0, convert forward delta via F/S
      (spot terms). Otherwise report library forward (discounted) delta and set
      ``delta_is_spot=False``.
    - Theta: per calendar day (py_vollib divides by 365).
    - Vega: per 1 vol point (py_vollib multiplies by 0.01).
    """
    if time_to_expiry <= 0:
        # Degenerate: intrinsic only; greeks not well-defined — zeros with flags.
        return BlackGreeks(
            price=forward_intrinsic(forward, strike, discount, right),
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            rho=0.0,
            delta_is_spot=spot is not None,
            theta_interpretively_limited=True,
        )

    r = rate_from_discount(discount, time_to_expiry)
    flag = _flag(right)
    px = float(_black_price(flag, forward, strike, time_to_expiry, r, sigma))
    d_fwd = float(_delta(flag, forward, strike, time_to_expiry, r, sigma))
    if spot is not None and spot > 0:
        d = d_fwd * (forward / spot)
        delta_is_spot = True
    else:
        d = d_fwd
        delta_is_spot = False

    limited = time_to_expiry * 365.0 < 1.0
    return BlackGreeks(
        price=px,
        delta=d,
        gamma=float(_gamma(flag, forward, strike, time_to_expiry, r, sigma)),
        vega=float(_vega(flag, forward, strike, time_to_expiry, r, sigma)),
        theta=float(_theta(flag, forward, strike, time_to_expiry, r, sigma)),
        rho=float(_rho(flag, forward, strike, time_to_expiry, r, sigma)),
        delta_is_spot=delta_is_spot,
        theta_interpretively_limited=limited,
    )


def price_grid(
    forwards: Any,
    strikes: Any,
    times: Any,
    discounts: Any,
    sigmas: Any,
    rights: Any,
    *,
    xp: Any = np,
) -> Any:
    """Vector-shaped price helper (xp-agnostic shell). Scalar loop for PR1."""
    f = xp.asarray(forwards, dtype=float)
    k = xp.asarray(strikes, dtype=float)
    t = xp.asarray(times, dtype=float)
    d = xp.asarray(discounts, dtype=float)
    s = xp.asarray(sigmas, dtype=float)
    # rights: expect array of 'c'/'p' or OptionRight — caller should pass flags
    out = xp.empty(f.shape, dtype=float)
    flat = zip(
        f.ravel(),
        k.ravel(),
        t.ravel(),
        d.ravel(),
        s.ravel(),
        xp.asarray(rights).ravel(),
        strict=True,
    )
    vals = []
    for fi, ki, ti, di, si, ri in flat:
        right = ri if isinstance(ri, OptionRight) else (
            OptionRight.CALL if str(ri).lower().startswith("c") else OptionRight.PUT
        )
        vals.append(price(float(fi), float(ki), float(ti), float(di), float(si), right))
    out = xp.asarray(vals, dtype=float).reshape(f.shape)
    return out
