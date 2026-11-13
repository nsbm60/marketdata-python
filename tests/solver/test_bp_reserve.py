"""bp_reserve_pct: the solve runs against bp_limit − reserve, leaving the reserve for the stock
repair (which may use up to the full bp_limit). Validation rejects a missing or out-of-range reserve.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import optimizer  # noqa: E402


def _opt(uid: str, theta: float, iv: float, rv: float, margin: float) -> dict:
    """One short ATM option (delta 0, gamma 0 -> no band or risk pressure) whose positive edge
    makes the solver sell as many as buying power allows."""
    return {
        "id": f"{uid}260101C00100000-short", "underlying": uid, "type": "option",
        "direction": "short", "current_quantity": 0,
        "theta_per_unit": theta, "exec_cost_per_unit": 0.0,
        "delta_per_unit": 0.0, "gamma_per_unit": 0.0, "margin_per_unit": margin,
        "iv": iv, "rv_forecast": rv, "days_to_expiry": 5.0, "strike": 100.0, "expiry": "2026-01-01",
    }


def _und(uid: str) -> dict:
    return {"underlying": uid, "spot": 100.0, "delta_target_pct": 0.0,
            "delta_tolerance_pct": 10.0, "tradeable": True, "sigma_rv": 0.4}


# bp_limit = 60% of 100k = 60,000; reserve 10% -> bp_solve_limit = 50,000.
POLICY = {"bp_limit_pct": 60, "bp_reserve_pct": 10, "risk_aversion": 0.0,
          "hedge_interval_days": 1.0, "min_improvement": 0.0}
CAPITAL = {"net_liq": 100_000.0}


def _bundle(dvs: list, policy: dict = POLICY) -> dict:
    groups = optimizer._group_by_underlying(dvs)
    return optimizer._extract_inputs(dvs, [_und(dv["underlying"]) for dv in dvs],
                                     policy, CAPITAL, groups, horizon_days=5.0)


# ── the solve respects bp_solve_limit, not the full bp_limit ────────────────────────────────────

def test_solve_stays_within_bp_solve_limit() -> None:
    dvs = [_opt("AAA", theta=1.0, iv=1.0, rv=0.5, margin=100.0)]  # edge = 1·(1−0.25) > 0
    bundle = _bundle(dvs)
    assert bundle["bp_solve_limit"] == 50_000.0 and bundle["bp_limit"] == 60_000.0
    target, problem, _ = optimizer._build_and_solve(len(dvs), bundle)
    assert optimizer._map_status(problem.status) == optimizer.STATUS_OPTIMAL
    used = float(bundle["margin"] @ target.value)
    assert used <= bundle["bp_solve_limit"] + 1e-6          # bound by the solve limit
    assert used < bundle["bp_limit"] - 1.0                  # the reserve is left unused
    assert used > bundle["bp_solve_limit"] - 100.0          # edge pushes it to the solve limit


# ── the reserve gives the repair room between the two limits ────────────────────────────────────

def _repair_bundle(bp_limit: float) -> dict:
    """Post-solve book: the option (index 0) uses 800 margin at q=1 (the solve limit); its net delta
    6 is out of the ±5 band, so the repair wants to short 1 share (100 margin) to reach 5."""
    return {
        "und_names": ["AAA"], "u_lo": np.array([-5.0]), "u_hi": np.array([5.0]),
        "G_delta": np.array([[6.0, 1.0, -1.0]]),
        "margin": np.array([800.0, 100.0, 100.0]),
        "equity_legs": {"AAA": {"long": 1, "short": 2}}, "tradeable": {"AAA": True},
        "bp_limit": bp_limit, "net_liq": 100_000.0, "pu": {"AAA": (100.0, 0.0, 5.0)},
    }


def test_repair_uses_the_reserve_between_the_two_limits() -> None:
    q0 = [1, 0, 0]  # option held -> book margin 800 (= a solve limit of 800)
    # At the solve limit (no reserve room): budget = 800 − 800 = 0 -> cannot move -> bp_capped.
    _q, capped = optimizer._repair_band_with_stock(q0, _repair_bundle(bp_limit=800.0))
    assert capped["AAA"]["outcome"] == "bp_capped" and capped["AAA"]["moved_shares"] == 0
    # With a 100 reserve (full bp_limit 900): budget = 900 − 800 = 100 -> short 1 -> repaired.
    _q2, ok = optimizer._repair_band_with_stock(q0, _repair_bundle(bp_limit=900.0))
    assert ok["AAA"] == {"outcome": "repaired"}


# ── validation: a missing or out-of-range reserve is rejected ───────────────────────────────────

def test_missing_reserve_is_rejected() -> None:
    dvs = [_opt("AAA", 1.0, 1.0, 0.5, 100.0)]
    policy = {k: v for k, v in POLICY.items() if k != "bp_reserve_pct"}
    with pytest.raises(ValueError, match="bp_reserve_pct"):
        _bundle(dvs, policy)


def test_out_of_range_reserve_is_rejected() -> None:
    dvs = [_opt("AAA", 1.0, 1.0, 0.5, 100.0)]
    for bad in (60, 65, -1):  # == bp_limit_pct, above it, and negative
        with pytest.raises(ValueError, match="bp_reserve_pct"):
            _bundle(dvs, {**POLICY, "bp_reserve_pct": bad})
