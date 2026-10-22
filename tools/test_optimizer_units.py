#!/usr/bin/env python3
"""
Unit tests for optimizer.py internals that the integration test can't reach
cheaply: the CVXPY-status → response-status mapping, and the timeout
feasible/omitted branch (recommendations present when a feasible point exists,
omitted when it doesn't). No service or solve required — pure function tests.

    python tools/test_optimizer_units.py
"""

import sys
import time
from types import SimpleNamespace

sys.path.insert(0, ".")

import numpy as np

import optimizer as opt


def test_map_status():
    cases = {
        "optimal": opt.STATUS_OPTIMAL,
        "optimal_inaccurate": opt.STATUS_OPTIMAL,
        "infeasible": opt.STATUS_INFEASIBLE,
        "infeasible_inaccurate": opt.STATUS_INFEASIBLE,
        "user_limit": opt.STATUS_TIMEOUT,
        "unbounded": opt.STATUS_INTERNAL_ERROR,
        "unbounded_inaccurate": opt.STATUS_INTERNAL_ERROR,
        "solver_error": opt.STATUS_INTERNAL_ERROR,
        "infeasible_or_unbounded": opt.STATUS_INTERNAL_ERROR,
        None: opt.STATUS_INTERNAL_ERROR,
    }
    for cvxpy_status, expected in cases.items():
        got = opt._map_status(cvxpy_status)
        assert got == expected, f"_map_status({cvxpy_status!r}) = {got}, expected {expected}"
    print(f"OK: _map_status maps {len(cases)} statuses correctly")


def _tiny_bundle_and_groups():
    """One short option on one underlying — enough to exercise response building."""
    dvs = [{
        "id": "O:ZZZ260515P00100000-short", "underlying": "ZZZ", "type": "option",
        "direction": "short", "current_quantity": 0, "delta_per_unit": 20.0,
        "theta_per_unit": 10.0, "gamma_per_unit": -1.0, "margin_per_unit": 1000.0,
        "exec_cost_per_unit": 5.0, "strike": 100.0, "expiry": "2026-05-15",
        "days_to_expiry": 4.25, "iv": 0.3,
    }]
    per_underlying = [{"underlying": "ZZZ", "spot": 100.0, "delta_target_pct": 0,
                       "delta_tolerance_pct": 1.0, "max_positions": 8, "tradeable": True}]
    policy = {"bp_limit_pct": 60, "delta_ceiling_threshold": 0.30,
              "lambda_delta_ceiling": 1.0, "lambda_gamma": 1.0, "lambda_spread": 1.0,
              "min_improvement": 0.0}
    capital = {"net_liq": 250000.0}
    groups = opt._group_by_underlying(dvs)
    bundle = opt._extract_inputs(dvs, per_underlying, policy, capital, groups, horizon_days=5)
    return dvs, bundle, groups


def _fake_problem(status="user_limit"):
    return SimpleNamespace(status=status, solver_stats=SimpleNamespace(num_iters=3))


def test_timeout_feasible_branch():
    dvs, bundle, groups = _tiny_bundle_and_groups()
    n = len(dvs)
    meta = {"request_id": "u-1"}

    # Timeout WITH a feasible point -> recommendations + score present.
    resp = opt._build_solved_response(
        meta, opt.STATUS_TIMEOUT, np.array([1.0]), _fake_problem(), dvs, bundle, groups,
        True, n, 4, time.monotonic())
    assert resp["status"] == opt.STATUS_TIMEOUT
    assert "recommendations" in resp and "score" in resp, "feasible timeout must carry recs+score"
    assert resp["binding_constraints"] == []
    assert "error" in resp, "timeout should carry an error message"
    print("OK: timeout + feasible -> recommendations and score present")

    # Timeout WITHOUT a feasible point -> recommendations + score omitted.
    resp = opt._build_solved_response(
        meta, opt.STATUS_TIMEOUT, None, _fake_problem(), dvs, bundle, groups,
        True, n, 4, time.monotonic())
    assert resp["status"] == opt.STATUS_TIMEOUT
    assert "recommendations" not in resp, "infeasible timeout must omit recommendations"
    assert "score" not in resp, "infeasible timeout must omit score"
    assert resp["binding_constraints"] == []
    assert "error" in resp
    print("OK: timeout + no feasible point -> recommendations and score omitted")


def test_optimal_branch_has_no_error():
    dvs, bundle, groups = _tiny_bundle_and_groups()
    n = len(dvs)
    resp = opt._build_solved_response(
        {"request_id": "u-2"}, opt.STATUS_OPTIMAL, np.array([1.0]), _fake_problem("optimal"),
        dvs, bundle, groups, True, n, 4, time.monotonic())
    assert resp["status"] == opt.STATUS_OPTIMAL
    assert "recommendations" in resp and "score" in resp
    assert "error" not in resp, "optimal response must not carry an error field"
    print("OK: optimal -> recs+score present, no error field")


def test_gamma_overlap_same_day_hedge():
    """A same-day long call (h = 0.25) hedging a 5-day short gets hedge credit only for
    the 0.25-day overlap: the cross term is 2·γ_s q_s·γ_l q_l·min(h_s, h_l), not √(h_s·h_l)."""
    def dv(name, direction, gamma, dte):
        return {"id": f"O:ZZZ{name}-{direction}", "underlying": "ZZZ", "type": "option",
                "direction": direction, "current_quantity": 0, "delta_per_unit": 0.0,
                "theta_per_unit": 0.0, "gamma_per_unit": gamma, "margin_per_unit": 1.0,
                "exec_cost_per_unit": 0.0, "days_to_expiry": dte}
    dvs = [dv("260519C00100000", "short", -2.0, 7.235),   # h = min(7.235, 5) = 5
           dv("260512C00100000", "long", 3.0, 0.25)]      # h = 0.25 (same day)
    per_underlying = [{"underlying": "ZZZ", "spot": 100.0, "delta_target_pct": 0,
                       "delta_tolerance_pct": 1.0, "max_positions": 8, "tradeable": True}]
    policy = {"bp_limit_pct": 60, "delta_ceiling_threshold": 0.30,
              "lambda_delta_ceiling": 1.0, "lambda_gamma": 0.7, "lambda_spread": 1.0,
              "min_improvement": 0.0}
    groups = opt._group_by_underlying(dvs)
    bundle = opt._extract_inputs(dvs, per_underlying, policy, {"net_liq": 250000.0}, groups, horizon_days=5)
    qs, ql = 3, 4
    xs, xl = -2.0 * qs, 3.0 * ql
    expected = -0.7 * (xs * xs * 5.0 + xl * xl * 0.25 + 2 * xs * xl * 0.25)
    sqrt_h_form = -0.7 * (xs * np.sqrt(5.0) + xl * np.sqrt(0.25)) ** 2
    got = float(opt._compute_contributions(bundle, [qs, ql])[opt.OC_GAMMA].sum())
    assert abs(got - expected) < 1e-9, f"gamma penalty {got} != overlap form {expected}"
    assert abs(got - sqrt_h_form) > 1.0, "overlap form should differ from the √h form here"
    print(f"OK: same-day hedge credited for 0.25 days only: penalty {got:.4f} (√h form would give {sqrt_h_form:.4f})")


def test_dollar_band_becomes_a_share_band():
    """(target_pct ± tolerance_pct)/100 · net_liq / spot: the same 10% band is 50 shares on
    a $500 stock and 2,500 on a $10 one; a −2% target shifts it."""
    def eq(u, direction):
        return {"id": f"{u}-equity-{direction}", "underlying": u, "type": "equity",
                "direction": direction, "current_quantity": 0,
                "delta_per_unit": 1.0 if direction == "long" else -1.0, "theta_per_unit": 0.0,
                "gamma_per_unit": 0.0, "margin_per_unit": 1.0, "exec_cost_per_unit": 0.0}
    dvs = [eq("AAA", "long"), eq("AAA", "short"), eq("BBB", "long"), eq("BBB", "short")]
    per_underlying = [
        {"underlying": "AAA", "spot": 500.0, "delta_target_pct": 0, "delta_tolerance_pct": 10,
         "max_positions": 8, "tradeable": True},
        {"underlying": "BBB", "spot": 10.0, "delta_target_pct": -2, "delta_tolerance_pct": 10,
         "max_positions": 8, "tradeable": True}]
    policy = {"bp_limit_pct": 60, "delta_ceiling_threshold": 0.30, "lambda_delta_ceiling": 1.0,
              "lambda_gamma": 1.0, "lambda_spread": 1.0, "min_improvement": 0.0}
    groups = opt._group_by_underlying(dvs)
    bundle = opt._extract_inputs(dvs, per_underlying, policy, {"net_liq": 250000.0}, groups, horizon_days=5)
    k = {u: i for i, u in enumerate(bundle["und_names"])}
    assert np.allclose([bundle["u_lo"][k["AAA"]], bundle["u_hi"][k["AAA"]]], [-50.0, 50.0])
    assert np.allclose([bundle["u_lo"][k["BBB"]], bundle["u_hi"][k["BBB"]]], [-3000.0, 2000.0])
    # A book outside the band reports it in % of net liq.
    q = [0, 60, 0, 0]  # 60 shares short AAA = −$30,000 = −12% of net liq
    breach = opt._breached_constraints(bundle, q)
    assert len(breach) == 1 and breach[0]["underlying"] == "AAA", breach
    assert abs(breach[0]["value_pct"] - (-12.0)) < 1e-9 and breach[0]["tolerance_pct"] == 10.0, breach
    print("OK: 10% band = ±50 shares at $500, [−3000, +2000] at $10 with a −2% target; breach in % of net liq")

    # Non-positive net liq or tolerance is malformed input.
    for capital, pu_patch in (({"net_liq": 0.0}, {}), ({"net_liq": 250000.0}, {"delta_tolerance_pct": 0})):
        bad = [dict(per_underlying[0], **pu_patch), per_underlying[1]]
        try:
            opt._extract_inputs(dvs, bad, policy, capital, groups, horizon_days=5)
        except ValueError as e:
            print(f"OK: rejected: {e}")
        else:
            raise AssertionError(f"expected ValueError for {capital} {pu_patch}")


def main():
    print("=" * 60)
    print("Optimizer Unit Tests")
    print("=" * 60)
    test_map_status()
    test_timeout_feasible_branch()
    test_optimal_branch_has_no_error()
    test_gamma_overlap_same_day_hedge()
    test_dollar_band_becomes_a_share_band()
    print("\n" + "=" * 60)
    print("UNIT TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
