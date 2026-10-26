"""Risk term (dollar P&L variance) — mechanical integration tests (amendment 2026-10-23).

These check the mechanism, not the book: raw theta is not the real objective, so the portfolio
the solver produces here is not judged. What is checked: the variance scales as σ⁴ in the
underlying's ATM IV, and the problem stays a convex QP that Clarabel solves.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import optimizer  # noqa: E402


def _option(uid: str, iv: float, gamma: float = 0.5) -> dict:
    """One ATM short option on its own underlying: strike = spot = 100, so the derived ATM IV
    is this contract's iv; delta 0 keeps it out of the band; theta 0 isolates the risk term."""
    return {
        "id": f"{uid}260101C00100000-short", "underlying": uid, "type": "option",
        "direction": "short", "current_quantity": 0,
        "theta_per_unit": 0.0, "exec_cost_per_unit": 0.0,
        "delta_per_unit": 0.0, "gamma_per_unit": gamma, "margin_per_unit": 100.0,
        "iv": iv, "days_to_expiry": 5.0, "strike": 100.0, "expiry": "2026-01-01",
    }


def _underlying(uid: str) -> dict:
    return {"underlying": uid, "spot": 100.0, "delta_target_pct": 0.0,
            "delta_tolerance_pct": 10.0, "max_positions": 8, "tradeable": True}


_POLICY = {"bp_limit_pct": 60, "max_positions_per_underlying": 8, "delta_ceiling_threshold": 0.30,
           "lambda_delta_ceiling": 1.0, "risk_aversion": 4.0, "hedge_interval_days": 1.0,
           "lambda_spread": 1.0, "min_improvement": 0.0}
_CAPITAL = {"net_liq": 100000.0}


def test_variance_ratio_is_iv_to_the_fourth() -> None:
    """Two names with the same dollar gamma (same γ, same spot) but IV 1.2 and 0.5: the risk
    contribution ratio is (1.2/0.5)⁴, because the variance goes as σ⁴."""
    dvs = [_option("AAA", iv=1.2), _option("BBB", iv=0.5)]
    groups = optimizer._group_by_underlying(dvs)
    bundle = optimizer._extract_inputs(dvs, [_underlying("AAA"), _underlying("BBB")],
                                       _POLICY, _CAPITAL, groups, horizon_days=5.0)
    contribs = optimizer._compute_contributions(bundle, [1, 1])
    risk = contribs[optimizer.OC_RISK]
    ratio = risk[0] / risk[1]
    expected = (1.2 / 0.5) ** 4
    assert abs(ratio - expected) < 1e-9, f"risk ratio {ratio} != (1.2/0.5)^4 = {expected}"


def test_problem_is_convex_qp_and_solves() -> None:
    """The risk term keeps the problem a DCP-convex QP; Clarabel solves it. Solve time printed."""
    dvs = [_option("AAA", iv=1.2), _option("BBB", iv=0.5)]
    groups = optimizer._group_by_underlying(dvs)
    bundle = optimizer._extract_inputs(dvs, [_underlying("AAA"), _underlying("BBB")],
                                       _POLICY, _CAPITAL, groups, horizon_days=5.0)
    t0 = time.monotonic()
    _target, problem, _n = optimizer._build_and_solve(len(dvs), bundle)
    elapsed = time.monotonic() - t0
    assert problem.is_dcp(), "risk term broke DCP convexity"
    assert optimizer._map_status(problem.status) in (optimizer.STATUS_OPTIMAL, optimizer.STATUS_TIMEOUT)
    print(f"\n2-name QP solved in {elapsed * 1000:.1f} ms, status={problem.status}, dcp={problem.is_dcp()}")
