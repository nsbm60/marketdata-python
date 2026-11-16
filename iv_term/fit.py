"""
The weighted quadratic smile fit and the status ladder (method atm_v1) — pure NumPy, no ClickHouse.

Per (underlying, session_date, expiry) group, ClickHouse hands us the size-weighted moments of the
qualifying rows (Σw, Σw·k … Σw·k⁴, Σw·iv, Σw·k·iv, Σw·k²·iv) plus a few summary stats. From those
alone we:

  - scale the fit variable, u = k / s with s = sqrt(Σw·k² / Σw) (the size-weighted RMS of k). ATM is
    k = 0, so the raw-k moments scale closed-form: Σw·uʲ = Σw·kʲ / sʲ. The scaling makes the near-ATM
    system (k is tiny, k² tinier) well-conditioned, so COND_MAX is meaningful and comparable across
    groups;
  - solve the 3×3 normal equations for iv = A + B·u + C·u² and map back to raw k: a = A, slope b = B/s,
    curvature c = C/s². a is unaffected by the scaling because u = 0 at k = 0;
  - classify the group by the status ladder.

Everything here is deterministic and unit-tested; it never touches a calendar, a clock or a client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import (
    COND_MAX, FIT_RANGE_SLACK, MIN_STRIKES, MIN_TRADES, STATUS_FIT_FAILED, STATUS_FIT_OUT_OF_RANGE,
    STATUS_OK, STATUS_ONE_SIDED, STATUS_TOO_FEW_STRIKES, STATUS_TOO_FEW_TRADES,
)


@dataclass(frozen=True)
class SmileAgg:
    """The size-weighted moments and summary stats for one group, as returned by the aggregation.
    `w` is trade size; `k = ln(strike/forward)`; the per-row iv/tte/forward are the trade-basis
    greeks. All sums are over the qualifying rows only (size > 0, in the band and the window)."""

    n_trades: int
    n_strikes: int
    contracts: float   # Σw
    sw: float          # Σw            (== contracts; kept explicit for the fit math)
    swk: float         # Σw·k
    swk2: float        # Σw·k²
    swk3: float        # Σw·k³
    swk4: float        # Σw·k⁴
    swv: float         # Σw·iv
    swkv: float        # Σw·k·iv
    swk2v: float       # Σw·k²·iv
    sw_tte: float      # Σw·tte
    sw_fwd: float      # Σw·forward
    k_min: float
    k_max: float
    iv_min: float
    iv_max: float


@dataclass(frozen=True)
class SmileFit:
    """The classified outcome for one group. atm_iv/total_var are set only on `ok`; slope/curvature
    on `ok` and `fit_out_of_range` (the fit ran); tte/forward are the size-weighted means and are
    always set (a group always has ≥1 qualifying row)."""

    status: str
    atm_iv: Optional[float]
    total_var: Optional[float]
    slope: Optional[float]
    curvature: Optional[float]
    tte: float
    forward: float


def _solve(agg: SmileAgg) -> Optional[Tuple[float, float, float]]:
    """Solve the size-weighted quadratic in the scaled variable u = k/s and return (a, b, c) in raw-k
    units, or None if the scaled system is singular / ill-conditioned (cond > COND_MAX). Called only
    once the group has ≥3 distinct strikes straddling the forward, so s > 0."""
    s2 = agg.swk2 / agg.sw  # = Σw·k² / Σw ; s = sqrt(s2) is the weighted RMS of k
    if not (s2 > 0.0) or not np.isfinite(s2):
        return None
    s = float(np.sqrt(s2))
    # Moments of u: Σw·uʲ = Σw·kʲ / sʲ. With this s, m0 == m2 (E_w[u²] = 1).
    m0 = agg.sw
    m1 = agg.swk / s
    m2 = agg.swk2 / s**2
    m3 = agg.swk3 / s**3
    m4 = agg.swk4 / s**4
    r0 = agg.swv
    r1 = agg.swkv / s
    r2 = agg.swk2v / s**2
    mat = np.array([[m0, m1, m2], [m1, m2, m3], [m2, m3, m4]], dtype=float)
    rhs = np.array([r0, r1, r2], dtype=float)
    cond = float(np.linalg.cond(mat))
    if not np.isfinite(cond) or cond > COND_MAX:
        return None
    try:
        coef = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coef)):
        return None
    a_cap, b_cap, c_cap = (float(x) for x in coef)
    # Map the scaled coefficients back to raw k: iv = A + B·(k/s) + C·(k/s)².
    return a_cap, b_cap / s, c_cap / s**2


def classify(agg: SmileAgg) -> SmileFit:
    """Apply the status ladder in order and return the classified fit. The first reason that applies
    wins: too_few_trades, too_few_strikes, one_sided, fit_failed, fit_out_of_range, else ok."""
    tte = agg.sw_tte / agg.sw
    forward = agg.sw_fwd / agg.sw

    def out(status: str, a: Optional[float] = None, b: Optional[float] = None,
            c: Optional[float] = None, total_var: Optional[float] = None) -> SmileFit:
        return SmileFit(status=status, atm_iv=a, total_var=total_var, slope=b, curvature=c,
                        tte=tte, forward=forward)

    if agg.n_trades < MIN_TRADES:
        return out(STATUS_TOO_FEW_TRADES)
    if agg.n_strikes < MIN_STRIKES:
        return out(STATUS_TOO_FEW_STRIKES)
    # One-sided: no strike below the forward, or none above (k = 0 is the forward).
    if agg.k_min >= 0.0 or agg.k_max <= 0.0:
        return out(STATUS_ONE_SIDED)

    solved = _solve(agg)
    if solved is None:
        return out(STATUS_FIT_FAILED)
    a, b, c = solved

    # ATM is near the smile's minimum, so a is often slightly below every observed IV — allow
    # FIT_RANGE_SLACK before rejecting. a must still be positive.
    lo = agg.iv_min * (1.0 - FIT_RANGE_SLACK)
    hi = agg.iv_max * (1.0 + FIT_RANGE_SLACK)
    if a <= 0.0 or a < lo or a > hi:
        return out(STATUS_FIT_OUT_OF_RANGE, b=b, c=c)

    return out(STATUS_OK, a=a, b=b, c=c, total_var=a * a * tte)
