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
        "spread_cost_per_unit": 5.0, "strike": 100.0, "expiry": "2026-05-15",
        "days_to_expiry": 5, "iv": 0.3,
    }]
    per_underlying = [{"underlying": "ZZZ", "spot": 100.0, "delta_target": 0,
                       "tolerance": 25, "max_positions": 8, "tradeable": True}]
    policy = {"bp_limit_pct": 60, "concentration_limit_pct": 30, "delta_ceiling_threshold": 0.30,
              "lambda_delta_ceiling": 1.0, "lambda_gamma": 1.0, "lambda_spread": 0.65}
    capital = {"net_liq": 250000.0}
    groups = opt._group_by_underlying(dvs)
    bundle = opt._extract_inputs(dvs, per_underlying, policy, capital, groups)
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


def main():
    print("=" * 60)
    print("Optimizer Unit Tests")
    print("=" * 60)
    test_map_status()
    test_timeout_feasible_branch()
    test_optimal_branch_has_no_error()
    print("\n" + "=" * 60)
    print("UNIT TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
