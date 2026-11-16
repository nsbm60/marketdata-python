"""The weighted quadratic fit and the status ladder (method atm_v1)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from iv_term import (
    STATUS_FIT_FAILED, STATUS_FIT_OUT_OF_RANGE, STATUS_OK, STATUS_ONE_SIDED, STATUS_TOO_FEW_STRIKES,
    STATUS_TOO_FEW_TRADES,
)
from iv_term.fit import SmileAgg, classify


def make_agg(ks: Sequence[float], ivs: Sequence[float], ws: Optional[Sequence[float]] = None,
             tte: float = 0.05, forward: float = 100.0,
             n_trades: Optional[int] = None, n_strikes: Optional[int] = None) -> SmileAgg:
    """Build the aggregation's size-weighted moments from explicit (k, iv, w) rows, exactly as the
    ClickHouse query would. n_trades/n_strikes may be overridden to exercise the ladder gates in
    isolation."""
    k = np.asarray(ks, dtype=float)
    iv = np.asarray(ivs, dtype=float)
    w = np.ones_like(k) if ws is None else np.asarray(ws, dtype=float)
    tte_arr = np.full_like(k, tte)
    fwd_arr = np.full_like(k, forward)
    return SmileAgg(
        n_trades=len(k) if n_trades is None else n_trades,
        n_strikes=len(set(np.round(k, 12).tolist())) if n_strikes is None else n_strikes,
        contracts=float(w.sum()),
        sw=float(w.sum()),
        swk=float((w * k).sum()),
        swk2=float((w * k**2).sum()),
        swk3=float((w * k**3).sum()),
        swk4=float((w * k**4).sum()),
        swv=float((w * iv).sum()),
        swkv=float((w * k * iv).sum()),
        swk2v=float((w * k**2 * iv).sum()),
        sw_tte=float((w * tte_arr).sum()),
        sw_fwd=float((w * fwd_arr).sum()),
        k_min=float(k.min()),
        k_max=float(k.max()),
        iv_min=float(iv.min()),
        iv_max=float(iv.max()),
    )


def quad(ks: Sequence[float], a: float, b: float, c: float) -> list[float]:
    return [a + b * k + c * k * k for k in ks]


# ── the fit recovers the truth ────────────────────────────────────────────────────────────────────

def test_exact_quadratic_is_recovered() -> None:
    a, b, c = 0.32, 0.4, 6.0
    ks = [-0.06, -0.03, -0.01, 0.02, 0.05]
    agg = make_agg(ks, quad(ks, a, b, c))
    fit = classify(agg)
    assert fit.status == STATUS_OK
    assert fit.atm_iv is not None and abs(fit.atm_iv - a) < 1e-9
    assert fit.slope is not None and abs(fit.slope - b) < 1e-7
    assert fit.curvature is not None and abs(fit.curvature - c) < 1e-6
    assert fit.total_var is not None and abs(fit.total_var - a * a * fit.tte) < 1e-12


def test_weighted_lopsided_strike_mix_still_true_at_zero() -> None:
    # Strikes bunched on the put wing with heavy weight there, a couple on the call wing. An exact
    # quadratic is recovered regardless of the weighting, so a (the value at k=0) is exact.
    a, b, c = 0.28, -0.5, 9.0
    ks = [-0.08, -0.07, -0.06, -0.05, 0.03, 0.05]
    ws = [50.0, 40.0, 30.0, 20.0, 2.0, 1.0]
    agg = make_agg(ks, quad(ks, a, b, c), ws=ws)
    fit = classify(agg)
    assert fit.status == STATUS_OK
    assert fit.atm_iv is not None and abs(fit.atm_iv - a) < 1e-9


def test_near_atm_pm_0_05_comes_back_ok() -> None:
    # Realistic near-ATM strikes, a mild skewed-convex smile. The scaling (u = k/s) keeps the system
    # well-conditioned at these tiny k, so it must classify ok.
    ks = [-0.05, -0.025, 0.0, 0.025, 0.05]
    ivs = quad(ks, 0.22, 0.3, 4.0)
    agg = make_agg(ks, ivs)
    fit = classify(agg)
    assert fit.status == STATUS_OK
    assert fit.atm_iv is not None and abs(fit.atm_iv - 0.22) < 1e-9


def test_symmetric_convex_pm_0_03_ok_with_a_below_iv_min() -> None:
    # A symmetric convex smile sampled only in the ±0.03 wings (MIN_STRIKES forces ≥3 distinct, so
    # four strikes across [-0.03, +0.03] with none at the money). The fitted vertex a sits BELOW
    # every observed IV; FIT_RANGE_SLACK = 0.2 must still accept it as ok.
    a, c = 0.40, 8.0
    ks = [-0.03, -0.015, 0.015, 0.03]
    ivs = quad(ks, a, 0.0, c)  # symmetric: b = 0
    agg = make_agg(ks, ivs, n_trades=20)  # many trades across the four wing strikes
    fit = classify(agg)
    assert fit.status == STATUS_OK
    assert fit.atm_iv is not None and abs(fit.atm_iv - a) < 1e-9
    assert fit.atm_iv < agg.iv_min  # the ATM value is genuinely below the lowest observed IV


# ── the status ladder, each reason and the order it is checked in ──────────────────────────────────

def test_too_few_trades_beats_everything_else() -> None:
    # A perfectly fittable, two-sided, many-strike smile — but only 4 trades. too_few_trades wins.
    ks = [-0.05, -0.02, 0.02, 0.05]
    agg = make_agg(ks, quad(ks, 0.3, 0.2, 5.0), n_trades=4)
    assert classify(agg).status == STATUS_TOO_FEW_TRADES


def test_too_few_strikes_beats_one_sided_and_fit() -> None:
    # 10 trades, two-sided, fittable — but only 2 distinct strikes. too_few_strikes wins.
    ks = [-0.03, 0.03]
    agg = make_agg(ks, quad(ks, 0.3, 0.0, 5.0), ws=[6.0, 4.0], n_trades=10, n_strikes=2)
    assert classify(agg).status == STATUS_TOO_FEW_STRIKES


def test_one_sided_when_no_strike_below_the_forward() -> None:
    ks = [0.01, 0.03, 0.05]  # k_min >= 0 → all strikes above the forward
    agg = make_agg(ks, quad(ks, 0.3, 0.2, 5.0), n_trades=20)
    assert classify(agg).status == STATUS_ONE_SIDED


def test_one_sided_when_no_strike_above_the_forward() -> None:
    ks = [-0.05, -0.03, -0.01]  # k_max <= 0 → all strikes below the forward
    agg = make_agg(ks, quad(ks, 0.3, 0.2, 5.0), n_trades=20)
    assert classify(agg).status == STATUS_ONE_SIDED


def test_fit_failed_on_a_singular_system() -> None:
    # Mass at only two distinct k (rank-2 moment matrix) but n_strikes reported as 3, so the earlier
    # gates pass and the solve is reached — the scaled 3×3 is singular → cond = inf > COND_MAX.
    ks = [-0.03, -0.03, -0.03, 0.03, 0.03]
    agg = make_agg(ks, quad(ks, 0.3, 0.1, 5.0), n_trades=20, n_strikes=3)
    assert classify(agg).status == STATUS_FIT_FAILED


def test_fit_out_of_range_when_a_is_far_below_iv_min() -> None:
    # Steep symmetric convex smile sampled only in the wings: the vertex a is ~47% below iv_min,
    # beyond the 20% slack → fit_out_of_range (slope/curvature still carried; atm_iv/total_var NULL).
    a, c = 0.01, 10.0
    ks = [-0.04, -0.03, 0.03, 0.04]
    agg = make_agg(ks, quad(ks, a, 0.0, c), n_trades=20)
    fit = classify(agg)
    assert fit.status == STATUS_FIT_OUT_OF_RANGE
    assert fit.atm_iv is None and fit.total_var is None
    assert fit.slope is not None and fit.curvature is not None


def test_fit_out_of_range_when_a_is_not_positive() -> None:
    # A smile whose fitted vertex is negative (synthetic) — a <= 0 → fit_out_of_range.
    a, c = -0.05, 50.0
    ks = [-0.04, -0.03, 0.03, 0.04]
    agg = make_agg(ks, quad(ks, a, 0.0, c), n_trades=20)
    fit = classify(agg)
    assert fit.status == STATUS_FIT_OUT_OF_RANGE
    assert fit.atm_iv is None


def test_tte_and_forward_are_size_weighted_and_always_present() -> None:
    # Even on a non-ok status, the size-weighted tte/forward are reported.
    agg = make_agg([0.01, 0.02, 0.03], [0.3, 0.3, 0.3], ws=[1.0, 1.0, 2.0],
                   tte=0.05, forward=100.0, n_trades=3)
    fit = classify(agg)  # too_few_trades
    assert fit.status == STATUS_TOO_FEW_TRADES
    assert abs(fit.tte - 0.05) < 1e-12
    assert abs(fit.forward - 100.0) < 1e-9
