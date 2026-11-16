"""Black-76 forward-form pricing and analytic greeks.

IV inversion is Jäckel's "Let's Be Rational" (native, greeks.solver.invert). The analytic greeks are
closed-form Black-76 in NumPy here — delta, gamma, theta, vega, rho — matching py_vollib's conventions
exactly (forward delta/gamma converted to spot terms by F/S and (F/S)², theta per calendar day, vega
and rho per 1% move). The standard normal CDF is scipy's ``ndtr`` (it agrees with vollib's Cody-based
norm_cdf to ~2.5e-15 relative over the whole range, well inside the 1e-9 greek tolerance).

``greeks_batch`` is the one implementation; the scalar ``greeks`` is a thin n=1 wrapper, so live and
history share one path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

from greeks.domain import OptionRight

# py_vollib re-exports vollib; the forward Black price helper stays on it (not a hot path).
from py_vollib.black import black as _black_price

_ONE_OVER_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


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
    """Analytic greeks at a given vol. Delta and gamma are spot terms when spot is set."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float  # per calendar day (py_vollib convention)
    rho: float
    delta_is_spot: bool
    theta_interpretively_limited: bool


@dataclass(frozen=True)
class BatchGreeks:
    """The arrays greeks_batch returns — one entry per input row, same conventions as BlackGreeks."""

    price: npt.NDArray[np.float64]
    delta: npt.NDArray[np.float64]
    gamma: npt.NDArray[np.float64]
    vega: npt.NDArray[np.float64]
    theta: npt.NDArray[np.float64]
    rho: npt.NDArray[np.float64]
    delta_is_spot: npt.NDArray[np.bool_]
    theta_interpretively_limited: npt.NDArray[np.bool_]


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
    return float(_black_price(_flag(right), forward, strike, time_to_expiry, r, sigma))


def greeks_batch(
    forward: npt.NDArray[np.float64],
    strike: npt.NDArray[np.float64],
    time_to_expiry: npt.NDArray[np.float64],
    discount: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
    is_call: npt.NDArray[np.bool_],
    spot: npt.NDArray[np.float64],
) -> BatchGreeks:
    """Closed-form Black-76 greeks for a batch of rows (py_vollib's analytic formulas, vectorized).

    ``spot`` carries NaN where no spot is given (scalar ``greeks(spot=None)``); such rows report forward
    (discounted) delta/gamma and ``delta_is_spot=False``. A row with ``time_to_expiry <= 0`` is
    degenerate: intrinsic price, zero greeks, ``theta_interpretively_limited=True`` — exactly the
    scalar function's behaviour.
    """
    F, K, t, D, s = forward, strike, time_to_expiry, discount, sigma
    spot_given = ~np.isnan(spot)
    spot_pos = spot_given & (spot > 0.0)
    pos = t > 0.0  # non-degenerate rows

    with np.errstate(all="ignore"):  # degenerate rows produce NaNs here, overwritten below
        t_safe = np.where(pos, t, 1.0)
        sqrt_t = np.sqrt(t_safe)
        r = np.where(pos, -np.log(D) / t_safe, 0.0)
        e_rt = np.exp(-r * t)
        d1 = np.where(pos, (np.log(F / K) + s * s * t / 2.0) / (s * sqrt_t), 0.0)
        d2 = d1 - s * sqrt_t
        pdf_d1 = _ONE_OVER_SQRT_2PI * np.exp(-0.5 * d1 * d1)
        n_d1, n_d2 = ndtr(d1), ndtr(d2)
        n_md1, n_md2 = ndtr(-d1), ndtr(-d2)

        delta_fwd = np.where(is_call, e_rt * n_d1, -e_rt * n_md1)
        gamma_fwd = pdf_d1 * e_rt / (F * s * sqrt_t)
        vega = F * e_rt * pdf_d1 * sqrt_t * 0.01
        first = F * e_rt * pdf_d1 * s / (2.0 * sqrt_t)
        theta = np.where(
            is_call,
            -(first - r * F * e_rt * n_d1 + r * K * e_rt * n_d2) / 365.0,
            (-first - r * F * e_rt * n_md1 + r * K * e_rt * n_md2) / 365.0,
        )
        black_px = e_rt * np.where(is_call, F * n_d1 - K * n_d2, K * n_md2 - F * n_md1)
        rho = -t * black_px * 0.01

        f_over_s = np.where(spot_pos, F / np.where(spot_pos, spot, 1.0), 1.0)
        delta = np.where(spot_pos, delta_fwd * f_over_s, delta_fwd)
        gamma = np.where(spot_pos, gamma_fwd * f_over_s * f_over_s, gamma_fwd)

    delta_is_spot = np.where(pos, spot_pos, spot_given)
    theta_limited = np.where(pos, t * 365.0 < 1.0, True)

    if (~pos).any():  # degenerate: intrinsic price, zero greeks
        intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0)) * D
        black_px = np.where(pos, black_px, intrinsic)
        zero = np.zeros_like(black_px)
        delta = np.where(pos, delta, zero)
        gamma = np.where(pos, gamma, zero)
        vega = np.where(pos, vega, zero)
        theta = np.where(pos, theta, zero)
        rho = np.where(pos, rho, zero)

    return BatchGreeks(
        price=black_px, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho,
        delta_is_spot=delta_is_spot, theta_interpretively_limited=theta_limited)


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
    """Analytic Black-76 greeks for one option — a thin n=1 wrapper over greeks_batch."""
    b = greeks_batch(
        np.array([forward], dtype=np.float64), np.array([strike], dtype=np.float64),
        np.array([time_to_expiry], dtype=np.float64), np.array([discount], dtype=np.float64),
        np.array([sigma], dtype=np.float64), np.array([right is OptionRight.CALL], dtype=bool),
        np.array([math.nan if spot is None else spot], dtype=np.float64))
    return BlackGreeks(
        price=float(b.price[0]), delta=float(b.delta[0]), gamma=float(b.gamma[0]),
        vega=float(b.vega[0]), theta=float(b.theta[0]), rho=float(b.rho[0]),
        delta_is_spot=bool(b.delta_is_spot[0]),
        theta_interpretively_limited=bool(b.theta_interpretively_limited[0]))


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
    out = xp.empty(f.shape, dtype=float)
    flat = zip(
        f.ravel(), k.ravel(), t.ravel(), d.ravel(), s.ravel(), xp.asarray(rights).ravel(),
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
