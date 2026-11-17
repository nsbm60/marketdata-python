"""
The weighted isotonic fit and horizon interpolation (method iso_v1) — pure NumPy, no ClickHouse.

Per (underlying, session_date), over the ok iv_term expiries with τ = trading days to expiry, raw
total variance and weight n_trades:

  - pool expiries sharing the same τ (weighted mean value, summed support), then run a weighted
    pool-adjacent-violators regression for a non-decreasing fit of total variance on τ;
  - a fixed anchor at (τ=0, 0) is NOT part of the fit (total variance ≥ 0, so the anchor never
    constrains it); it serves only as the lower bracket for interpolation;
  - expiry points carry τ, raw, fitted (their pooled block's value) and support (Σ n_trades of the
    block); horizon points for HORIZONS_TD interpolate linearly in τ between bracketing fitted
    points, support = the smaller bracket's (the anchor is unlimited), status beyond_last_expiry past
    the last expiry with no extrapolation.

Deterministic and unit-tested; never touches a calendar, clock or client.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import HORIZONS_TD, KIND_EXPIRY, KIND_HORIZON, STATUS_BEYOND_LAST_EXPIRY, STATUS_OK


@dataclass(frozen=True)
class ExpiryInput:
    """One ok iv_term expiry for a (underlying, session_date): τ trading days to expiry, raw total
    variance, and the trade-count weight."""

    expiry: date
    tau: int
    raw_total_var: float
    n_trades: float


@dataclass(frozen=True)
class CurvePoint:
    """One output row. kind 'expiry' carries expiry/raw/fit/support; kind 'horizon' carries
    horizon_td/status/fit/support and the bracketing expiries (lo None when the anchor brackets)."""

    kind: str
    tau_td: int
    status: str
    expiry: Optional[date]
    horizon_td: Optional[int]
    raw_total_var: Optional[float]
    fit_total_var: Optional[float]
    support: Optional[float]
    bracket_lo_expiry: Optional[date]
    bracket_hi_expiry: Optional[date]


@dataclass(frozen=True)
class _Node:
    """An interpolation node on the fitted curve: a τ with the fitted value, the pooled-block support
    and the representative expiry (None for the anchor)."""

    tau: int
    value: float
    support: float
    expiry: Optional[date]


def pava(values: Sequence[float], weights: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Weighted pool-adjacent-violators for a non-decreasing fit. `values`/`weights` are ordered by x
    ascending (ties already pooled). Returns (fitted value per input point, block weight per input
    point) — the block weight is the sum of the pooled weights, i.e. the block's support."""
    block_val: List[float] = []
    block_wt: List[float] = []
    block_cnt: List[int] = []
    for v, w in zip(values, weights):
        block_val.append(float(v))
        block_wt.append(float(w))
        block_cnt.append(1)
        # Pool the last two blocks while they violate monotonicity (prev value > this value).
        while len(block_val) > 1 and block_val[-2] > block_val[-1]:
            w2 = block_wt[-2] + block_wt[-1]
            merged = ((block_val[-2] * block_wt[-2] + block_val[-1] * block_wt[-1]) / w2
                      if w2 > 0 else 0.5 * (block_val[-2] + block_val[-1]))
            cnt2 = block_cnt[-2] + block_cnt[-1]
            block_val[-2:] = [merged]
            block_wt[-2:] = [w2]
            block_cnt[-2:] = [cnt2]
    fitted: List[float] = []
    support: List[float] = []
    for v, w, c in zip(block_val, block_wt, block_cnt):
        fitted.extend([v] * c)
        support.extend([w] * c)
    return fitted, support


def _pool_by_tau(points: Sequence[ExpiryInput]) -> List[Tuple[int, float, float, date]]:
    """Pool expiries sharing a τ: (τ, weighted-mean value, summed support, representative expiry),
    sorted by τ. The representative is the expiry with the largest support (ties → earliest)."""
    by_tau: dict[int, List[ExpiryInput]] = {}
    for p in points:
        by_tau.setdefault(p.tau, []).append(p)
    out: List[Tuple[int, float, float, date]] = []
    for tau in sorted(by_tau):
        members = by_tau[tau]
        wsum = sum(m.n_trades for m in members)
        value = (sum(m.raw_total_var * m.n_trades for m in members) / wsum
                 if wsum > 0 else sum(m.raw_total_var for m in members) / len(members))
        rep = max(members, key=lambda m: (m.n_trades, -m.expiry.toordinal())).expiry
        out.append((tau, value, wsum, rep))
    return out


def build_curve(points: Sequence[ExpiryInput]) -> List[CurvePoint]:
    """Build all curve rows for one (underlying, session_date). Empty input → no rows (the caller
    still writes a ledger row recording 0)."""
    if not points:
        return []

    pooled = _pool_by_tau(points)
    taus = [t for (t, _, _, _) in pooled]
    values = [v for (_, v, _, _) in pooled]
    weights = [w for (_, _, w, _) in pooled]
    reps = [r for (_, _, _, r) in pooled]
    fitted, support = pava(values, weights)
    fit_by_tau = dict(zip(taus, fitted))
    support_by_tau = dict(zip(taus, support))

    rows: List[CurvePoint] = []

    # One expiry row per ok expiry: its own raw, its block's fitted value and support.
    for p in sorted(points, key=lambda x: (x.tau, x.expiry)):
        rows.append(CurvePoint(
            kind=KIND_EXPIRY, tau_td=p.tau, status=STATUS_OK, expiry=p.expiry, horizon_td=None,
            raw_total_var=p.raw_total_var, fit_total_var=fit_by_tau[p.tau],
            support=support_by_tau[p.tau], bracket_lo_expiry=None, bracket_hi_expiry=None))

    # Interpolation nodes: the fixed anchor (τ=0, 0, unlimited support) then the fitted τ-nodes.
    nodes: List[_Node] = [_Node(tau=0, value=0.0, support=math.inf, expiry=None)]
    for tau, rep in zip(taus, reps):
        nodes.append(_Node(tau=tau, value=fit_by_tau[tau], support=support_by_tau[tau], expiry=rep))

    for horizon in HORIZONS_TD:
        rows.append(_horizon_row(nodes, horizon))
    return rows


def _horizon_row(nodes: List[_Node], horizon: int) -> CurvePoint:
    last = nodes[-1]
    if horizon > last.tau:  # beyond the last ok expiry — no extrapolation
        return CurvePoint(
            kind=KIND_HORIZON, tau_td=horizon, status=STATUS_BEYOND_LAST_EXPIRY, expiry=None,
            horizon_td=horizon, raw_total_var=None, fit_total_var=None, support=None,
            bracket_lo_expiry=None, bracket_hi_expiry=None)

    # Lower bracket = last node with τ <= H; upper = first node with τ >= H.
    lo = max((n for n in nodes if n.tau <= horizon), key=lambda n: n.tau)
    hi = min((n for n in nodes if n.tau >= horizon), key=lambda n: n.tau)
    if hi.tau == lo.tau:                     # H lands exactly on a node
        fit = lo.value
    else:
        frac = (horizon - lo.tau) / (hi.tau - lo.tau)
        fit = lo.value + (hi.value - lo.value) * frac
    support = min(lo.support, hi.support)    # anchor support is inf, so the finite bracket wins
    return CurvePoint(
        kind=KIND_HORIZON, tau_td=horizon, status=STATUS_OK, expiry=None, horizon_td=horizon,
        raw_total_var=None, fit_total_var=fit, support=support,
        bracket_lo_expiry=lo.expiry, bracket_hi_expiry=hi.expiry)
