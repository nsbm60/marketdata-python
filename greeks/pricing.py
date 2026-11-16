"""The shared option-pricing core (greeks-server-spec.md §2, §5.4, §7). Pure.

Given a price and a spot — whatever their basis: a two-sided quote mid for the live server, a trade
print for the historical job — invert IV and the greeks under Black-76: carry (forward/discount/TTE)
from the SOFR rate and dividends, then the native "Let's Be Rational" solver. Live (greeks_server) and
historical (greeks.history) share THIS ONE core; there is no second inversion path. ``price_options``
is the batch form — one carry per row (memoized instants), then ONE invert_batch and ONE greeks_batch
over the live rows; ``price_option`` is the n=1 wrapper. Only the price basis differs, and the caller
records it. Nothing is defaulted — a failure carries the solver's reason with the greeks left absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, List, Optional, Sequence

import numpy as np

from greeks.domain import Dividend, FailureReason, OptionRight, RowStatus
from greeks.forwards.carry import carry
from greeks.solver.black76 import greeks_batch
from greeks.solver.invert import invert_batch

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class OptionInput:
    """One option to price. ``session_date``, ``methodology_version``, ``t_floor_minutes`` and
    ``expiry_time_et`` are shared across a ``price_options`` call, so they are not here."""
    expiry: date
    right: OptionRight
    strike: float
    price: float
    spot: float
    as_of: datetime
    rate_for: Callable[[date], float]
    dividends: Sequence[Dividend]


def _fail(reason: FailureReason, forward: Optional[float], discount: Optional[float],
          tte: Optional[float]) -> PricedGreeks:
    return PricedGreeks(RowStatus.FAILURE, reason, None, None, None, None, None, None, forward, discount, tte)


def price_options(
    inputs: Sequence[OptionInput],
    *,
    session_date: date,
    methodology_version: str,
    t_floor_minutes: float,
    expiry_time_et: str,
) -> List[PricedGreeks]:
    """Batch form of :func:`price_option`. Per option, in this order: an expiry before ``session_date``
    is T_BELOW_FLOOR with no carry; otherwise carry (per row, memoized instants); then ONE invert_batch
    over every row still live, then ONE greeks_batch. Returns one PricedGreeks per input, in input
    order, with exactly ``price_option``'s fields and failure semantics (forward/discount/tte set on a
    solver-stage failure, absent on an expiry failure; theta_limited carried).

    A carry exception on one row is isolated to that row as SOLVER_FAILED and logged at DEBUG — the
    per-row isolation lives here in the core, not in callers.
    """
    out: List[Optional[PricedGreeks]] = [None] * len(inputs)
    # Rows that reach the solver, staged into parallel lists for one batch call.
    s_price: List[float] = []
    s_strike: List[float] = []
    s_call: List[bool] = []
    s_fwd: List[float] = []
    s_disc: List[float] = []
    s_tte: List[float] = []
    s_spot: List[float] = []
    staged: List[int] = []

    for i, inp in enumerate(inputs):
        if inp.expiry < session_date:
            out[i] = _fail(FailureReason.T_BELOW_FLOOR, None, None, None)
            continue
        try:
            c = carry(inp.spot, inp.rate_for(inp.expiry), inp.as_of, inp.expiry, inp.dividends,
                      expiry_time_et=expiry_time_et)
        except Exception as e:  # noqa: BLE001 — per-row isolation: one bad row fails alone
            log.debug("price_options: carry raised %s: %s — solver_failed", type(e).__name__, e)
            out[i] = _fail(FailureReason.SOLVER_FAILED, None, None, None)
            continue
        s_price.append(inp.price)
        s_strike.append(inp.strike)
        s_call.append(inp.right is OptionRight.CALL)
        s_fwd.append(c.forward)
        s_disc.append(c.discount)
        s_tte.append(c.time_to_expiry)
        s_spot.append(inp.spot)
        staged.append(i)

    if staged:
        fwd = np.array(s_fwd, dtype=np.float64)
        strike = np.array(s_strike, dtype=np.float64)
        tte = np.array(s_tte, dtype=np.float64)
        disc = np.array(s_disc, dtype=np.float64)
        is_call = np.array(s_call, dtype=bool)
        iv, status, reason = invert_batch(
            np.array(s_price, dtype=np.float64), strike, is_call, fwd, disc, tte,
            methodology_version=methodology_version, t_floor_minutes=t_floor_minutes)
        # greeks for every staged row at its IV (failed rows get a dummy sigma; their greeks are unused).
        sigma = np.where(np.isfinite(iv) & (iv > 0), iv, 1.0)
        bg = greeks_batch(fwd, strike, tte, disc, sigma, is_call, np.array(s_spot, dtype=np.float64))
        # Bulk-convert the result arrays to Python lists once (C-level), so the per-row loop below is
        # plain float/bool indexing — not N×7 numpy-scalar → float conversions.
        iv_l = iv.tolist()
        delta_l, gamma_l, theta_l = bg.delta.tolist(), bg.gamma.tolist(), bg.theta.tolist()
        vega_l, rho_l = bg.vega.tolist(), bg.rho.tolist()
        tlim_l = bg.theta_interpretively_limited.tolist()
        for j, i in enumerate(staged):
            f_, d_, t_ = s_fwd[j], s_disc[j], s_tte[j]
            if status[j] is RowStatus.SUCCESS:
                out[i] = PricedGreeks(RowStatus.SUCCESS, None, iv_l[j], delta_l[j], gamma_l[j],
                                      theta_l[j], vega_l[j], rho_l[j], f_, d_, t_, tlim_l[j])
            else:
                out[i] = _fail(reason[j], f_, d_, t_)

    return [p for p in out if p is not None]


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
    """Invert IV + greeks for one option — a thin n=1 call to :func:`price_options`, whose docstring
    gives the fields and failure semantics. ``rate_for`` and ``dividends`` are for ``session_date``."""
    return price_options(
        [OptionInput(expiry=expiry, right=right, strike=strike, price=price, spot=spot, as_of=as_of,
                     rate_for=rate_for, dividends=dividends)],
        session_date=session_date, methodology_version=methodology_version,
        t_floor_minutes=t_floor_minutes, expiry_time_et=expiry_time_et)[0]
