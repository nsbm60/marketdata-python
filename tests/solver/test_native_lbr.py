"""The native "Let's Be Rational" core vs the former pure-Python path, on a large synthetic grid.

OLD = the pre-change path: the same pre-checks, py_vollib's implied_volatility (pure-Python LBR) and
py_vollib's analytic greeks. NEW = greeks.solver.invert_batch (Jäckel's reference C++) + greeks_batch
(NumPy/scipy). On every case: identical status and reason; IV to 1e-10 relative; greeks to 1e-9.

Note on the grid: it spans deep ITM to deep OTM, T from one minute to two years, vol 5–300%, calls and
puts, but is restricted to |ln(F/K)| / (σ√T) ≤ 8 — eight standard deviations is already far deeper than
any option trades, and beyond it the inversion is so ill-conditioned that py_vollib's OLDER ported LBR
and Jäckel's current reference diverge (the native being the more accurate). Real trades, and the
pre-checks, live well inside this. The reason codes are exercised on crafted invalid rows below.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from py_vollib.black.greeks import analytical as va
from py_vollib.black.implied_volatility import implied_volatility as vollib_iv

from greeks.domain import FailureReason, OptionRight, RowStatus
from greeks.solver.black76 import greeks_batch
from greeks.solver.invert import invert_batch
from vollib.black import undiscounted_black

_ABS_EPS = 1e-10


def _old_reason(price, strike, is_call, forward, discount, t, *, t_floor_minutes=15.0, tick=0.01):
    """The former scalar invert's pre-check ladder — first failing check wins; None if it reaches the
    solver and inverts to a positive IV."""
    flag = "c" if is_call else "p"
    if t < (t_floor_minutes * 60.0) / (365.0 * 86400.0):
        return FailureReason.T_BELOW_FLOOR
    if not math.isfinite(price) or price < 0:
        return FailureReason.SOLVER_FAILED
    if not math.isfinite(forward) or forward <= 0:
        return FailureReason.MISSING_FORWARD_INPUTS
    if not math.isfinite(strike) or strike <= 0:
        return FailureReason.MISSING_FORWARD_INPUTS
    if not math.isfinite(discount) or discount <= 0:
        return FailureReason.MISSING_FORWARD_INPUTS
    intrinsic = (max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)) * discount
    upper = (forward if is_call else strike) * discount
    if price < intrinsic - _ABS_EPS:
        return FailureReason.BELOW_INTRINSIC
    if price > upper + _ABS_EPS:
        return FailureReason.ABOVE_NO_ARB
    if t < 1.0 / 365.0 and price <= intrinsic + tick + _ABS_EPS:
        return FailureReason.QUANTIZED_NEAR_INTRINSIC
    try:
        r = -math.log(discount) / t
        iv = float(vollib_iv(price, forward, strike, r, t, flag))
    except Exception:  # noqa: BLE001
        return FailureReason.SOLVER_FAILED
    if not math.isfinite(iv) or iv <= 0:
        return FailureReason.SOLVER_FAILED
    return None  # reached the solver and inverted to a positive IV


def _old_iv_greeks(price, strike, is_call, forward, discount, t, spot):
    """OLD IV + greeks for a success row (py_vollib), with the former black76 F/S spot conversion."""
    flag = "c" if is_call else "p"
    r = -math.log(discount) / t
    iv = float(vollib_iv(price, forward, strike, r, t, flag))
    d_fwd = float(va.delta(flag, forward, strike, t, r, iv))
    g_fwd = float(va.gamma(flag, forward, strike, t, r, iv))
    f_over_s = forward / spot
    return {
        "iv": iv,
        "delta": d_fwd * f_over_s,
        "gamma": g_fwd * f_over_s * f_over_s,
        "vega": float(va.vega(flag, forward, strike, t, r, iv)),
        "theta": float(va.theta(flag, forward, strike, t, r, iv)),
        "rho": float(va.rho(flag, forward, strike, t, r, iv)),
    }


def test_native_matches_the_pure_python_path_on_a_large_grid() -> None:
    rng = np.random.default_rng(20261115)
    target = 100_000
    F_l, K_l, T_l, sig_l, call_l = [], [], [], [], []
    while len(F_l) < target:
        m = target * 2
        F = np.exp(rng.uniform(np.log(5.0), np.log(500.0), m))
        K = F * np.exp(rng.uniform(-1.5, 1.5, m))               # deep ITM to deep OTM
        T = np.exp(rng.uniform(np.log(1.0 / 365 / 24 / 60), np.log(2.0), m))  # 1 minute to 2 years
        sig = rng.uniform(0.05, 3.0, m)                         # 5% to 300%
        call = rng.integers(0, 2, m).astype(bool)
        keep = np.abs(np.log(F / K)) / (sig * np.sqrt(T)) <= 8.0
        F_l += list(F[keep]); K_l += list(K[keep]); T_l += list(T[keep])
        sig_l += list(sig[keep]); call_l += list(call[keep])
    n = target
    F = np.array(F_l[:n]); K = np.array(K_l[:n]); T = np.array(T_l[:n])
    sig = np.array(sig_l[:n]); is_call = np.array(call_l[:n], dtype=bool)
    disc = np.exp(-0.04 * T)
    und = np.array([undiscounted_black(float(F[i]), float(K[i]), float(sig[i]), float(T[i]),
                                       "c" if is_call[i] else "p") for i in range(n)])
    price = und * disc  # discounted market premium

    iv_new, status_new, reason_new = invert_batch(
        price, K, is_call, F, disc, T, methodology_version="v2")
    bg = greeks_batch(F, K, T, disc, np.where(np.isfinite(iv_new), iv_new, 1.0), is_call, F)

    # OLD path per row: the pre-check reason, and (on success) IV + greeks.
    iv_old = np.full(n, np.nan)
    g_old = {k: np.full(n, np.nan) for k in ("delta", "gamma", "vega", "theta", "rho")}
    succ = np.zeros(n, dtype=bool)
    for i in range(n):
        rc = _old_reason(float(price[i]), float(K[i]), bool(is_call[i]), float(F[i]),
                         float(disc[i]), float(T[i]))
        # identical status and reason on every case
        assert reason_new[i] == rc, f"row {i}: reason {reason_new[i]} != {rc}"
        assert (status_new[i] is RowStatus.SUCCESS) == (rc is None)
        if rc is None:
            succ[i] = True
            r = _old_iv_greeks(float(price[i]), float(K[i]), bool(is_call[i]), float(F[i]),
                               float(disc[i]), float(T[i]), float(F[i]))
            iv_old[i] = r["iv"]
            for k in g_old:
                g_old[k][i] = r[k]

    assert succ.sum() > n // 2, f"too few successes to validate ({succ.sum()})"
    iv_rel = np.abs(iv_new[succ] - iv_old[succ]) / np.abs(iv_old[succ])
    assert iv_rel.max() < 1e-10, f"max IV rel diff {iv_rel.max():.3e}"
    for k, new in (("delta", bg.delta), ("gamma", bg.gamma), ("vega", bg.vega),
                   ("theta", bg.theta), ("rho", bg.rho)):
        old = g_old[k][succ]
        denom = np.maximum(np.abs(old), 1e-12)  # abs floor: near-zero greeks (deep OTM) aren't rel-meaningful
        rel = np.abs(new[succ] - old) / denom
        assert rel.max() < 1e-9, f"max {k} rel diff {rel.max():.3e}"


def test_reason_codes_match_the_pure_python_ladder() -> None:
    # Crafted rows, one per reason plus successes, checked against the OLD pre-check ladder.
    F = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    K = np.array([100.0, 100.0,  50.0, 100.0,  50.0, 100.0], dtype=np.float64)
    is_call = np.array([True, True, True, True, True, True])
    disc = np.exp(-0.04 * np.array([0.5, 0.0001, 0.5, 0.5, 0.0005, 0.5]))
    T = np.array([0.5, 0.0001, 0.5, 0.5, 0.0005, 0.5], dtype=np.float64)
    # prices: ok, t_below_floor(t tiny), below_intrinsic(0), above_no_arb(>F*disc), quantized, ok-put-ish
    price = np.array([8.0, 8.0, 0.0, 1e9, 50.0 * disc[4] + 0.005, 8.0], dtype=np.float64)
    iv, status, reason = invert_batch(price, K, is_call, F, disc, T, methodology_version="v2")
    for i in range(len(F)):
        exp = _old_reason(float(price[i]), float(K[i]), bool(is_call[i]), float(F[i]),
                          float(disc[i]), float(T[i]))
        got = reason[i]
        assert got == exp, f"row {i}: got {got}, expected {exp}"
        assert (status[i] is RowStatus.SUCCESS) == (exp is None)
