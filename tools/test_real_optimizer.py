#!/usr/bin/env python3
"""
Integration harness for the real optimizer service (optimizer.py).

The continuous guard rail for the integration-first build per
pc-solver-spec-draft.md. Adapted from test_shell_optimizer.py with three
changes for the real solver: per-underlying records carry `spot` (needed by
the concentration constraint), realistic margins so the BP constraint can
bind, and solver_name is asserted to be the configured backend.

Start the real optimizer in one terminal:
    python optimizer.py

Run this test in another:
    python tools/test_real_optimizer.py

LAYER 2 scope: protocol round-trip + full arithmetic sanity for the complete
formulation (penalties, hard constraints, post-solve cardinality). The gamma
penalty is independently recomputed from the response to verify the
per-underlying group-sum attribution. There is no yield floor — yield is a
display measure, not a control — so no contract is pre-filtered; a vestigial
yield_floor_pct in policy must be ignored. The full status taxonomy and
binding_constraints get their own assertions in Layer 3.
"""

import json
import sys
import uuid
from collections import defaultdict

sys.path.insert(0, ".")

import zmq
from discovery.service_locator import ServiceLocator

EXPECTED_SOLVER = "clarabel"  # default backend; override if OPTIMIZER_SOLVER is set
TOL = 1e-4  # arithmetic-sanity tolerance


def make_request(verbose=False):
    """Synthetic request: multiple underlyings, held + candidate + closeable_only,
    short (theta>0) and long (theta<0) directions, an equity leg, spot per
    underlying. Margins realistic so BP can bind. policy.yield_floor_pct is set
    deliberately high (0.5) to confirm the solver ignores it (vestigial field)."""
    dvs = [
        {"id": "O:NVDA260515P00215000-short", "underlying": "NVDA", "type": "option",
         "direction": "short", "current_quantity": 2, "delta_per_unit": 27.5,
         "theta_per_unit": 12.5, "gamma_per_unit": -1.8, "margin_per_unit": 4327.0,
         "premium_per_unit": 280.0, "spread_cost_per_unit": 8.0, "iv": 0.319,
         "days_to_expiry": 3, "strike": 215.0, "expiry": "2026-05-15"},
        {"id": "O:NVDA260515P00220000-short", "underlying": "NVDA", "type": "option",
         "direction": "short", "current_quantity": 0, "delta_per_unit": 35.0,
         "theta_per_unit": 15.2, "gamma_per_unit": -2.1, "margin_per_unit": 4527.0,
         "premium_per_unit": 320.0, "spread_cost_per_unit": 9.0, "iv": 0.325,
         "days_to_expiry": 3, "strike": 220.0, "expiry": "2026-05-15"},
        {"id": "O:NVDA260515P00215000-long", "underlying": "NVDA", "type": "option",
         "direction": "long", "current_quantity": 0, "delta_per_unit": -27.5,
         "theta_per_unit": -12.5, "gamma_per_unit": 1.8, "margin_per_unit": 280.0,
         "premium_per_unit": -280.0, "spread_cost_per_unit": 8.0, "iv": 0.319,
         "days_to_expiry": 3, "strike": 215.0, "expiry": "2026-05-15"},
        {"id": "O:MU260515C00120000-short", "underlying": "MU", "type": "option",
         "direction": "short", "current_quantity": 0, "delta_per_unit": -22.0,
         "theta_per_unit": 9.8, "gamma_per_unit": -1.1, "margin_per_unit": 2200.0,
         "premium_per_unit": 150.0, "spread_cost_per_unit": 6.0, "iv": 0.41,
         "days_to_expiry": 5, "strike": 120.0, "expiry": "2026-05-15"},
        {"id": "O:TSLA260515P00250000-short", "underlying": "TSLA", "type": "option",
         "direction": "short", "current_quantity": 2, "closeable_only": True,
         "delta_per_unit": 18.0, "theta_per_unit": 8.4, "gamma_per_unit": -1.2,
         "margin_per_unit": 4980.0, "premium_per_unit": 175.0, "spread_cost_per_unit": 11.0,
         "iv": 0.412, "days_to_expiry": 3, "strike": 250.0, "expiry": "2026-05-15"},
        {"id": "NVDA-equity-long", "underlying": "NVDA", "type": "equity",
         "direction": "long", "current_quantity": 100, "delta_per_unit": 1.0,
         "theta_per_unit": 0, "gamma_per_unit": 0, "margin_per_unit": 21.65,
         "premium_per_unit": 216.49, "spread_cost_per_unit": 0.01},
    ]
    return {
        "schema_version": 1,
        "request_id": f"test-{uuid.uuid4().hex[:8]}",
        "policy_version": "v3",
        "timestamp": "2026-05-13T12:00:00Z",
        "verbose": verbose,
        "decision_variables": dvs,
        "per_underlying": [
            {"underlying": "NVDA", "spot": 215.0, "delta_target": 0, "tolerance": 25, "max_positions": 8, "tradeable": True},
            {"underlying": "MU", "spot": 118.0, "delta_target": 0, "tolerance": 25, "max_positions": 8, "tradeable": True},
            {"underlying": "TSLA", "spot": 250.0, "delta_target": 0, "tolerance": 25, "max_positions": 8, "tradeable": False},
        ],
        "policy": {
            "bp_limit_pct": 60, "max_positions_per_underlying": 8, "concentration_limit_pct": 30,
            "yield_floor_pct": 0.5, "delta_ceiling_threshold": 0.30,
            "lambda_delta_ceiling": 1.0, "lambda_gamma": 1.0, "lambda_spread": 0.65,
            "default_delta_tolerance": 25.0,
        },
        "capital": {"net_liq": 250000.0, "current_bp_usage": 45000.0, "current_margin": 32000.0},
    }


def make_infeasible_request():
    """Single positive-delta short with a large NEGATIVE delta target. Net delta
    is >= 0 for any non-negative quantity, so the target band [-100001, -99999]
    is unreachable — a deterministic infeasibility."""
    return {
        "schema_version": 1,
        "request_id": f"test-inf-{uuid.uuid4().hex[:8]}",
        "policy_version": "v3",
        "timestamp": "2026-05-13T12:00:00Z",
        "verbose": False,
        "decision_variables": [{
            "id": "O:ZZZ260515P00100000-short", "underlying": "ZZZ", "type": "option",
            "direction": "short", "current_quantity": 0, "delta_per_unit": 30.0,
            "theta_per_unit": 10.0, "gamma_per_unit": -1.0, "margin_per_unit": 1000.0,
            "premium_per_unit": 100.0, "spread_cost_per_unit": 5.0, "iv": 0.3,
            "days_to_expiry": 5, "strike": 100.0, "expiry": "2026-05-15"}],
        "per_underlying": [
            {"underlying": "ZZZ", "spot": 100.0, "delta_target": -100000, "tolerance": 1,
             "max_positions": 8, "tradeable": True}],
        "policy": {
            "bp_limit_pct": 60, "max_positions_per_underlying": 8, "concentration_limit_pct": 30,
            "yield_floor_pct": 0.001, "delta_ceiling_threshold": 0.30,
            "lambda_delta_ceiling": 1.0, "lambda_gamma": 1.0, "lambda_spread": 0.65,
            "default_delta_tolerance": 25.0},
        "capital": {"net_liq": 250000.0, "current_bp_usage": 0.0, "current_margin": 0.0},
    }


def send_request(dealer, request):
    dealer.send_multipart([b"", json.dumps({"op": "optimize", **request}).encode()])
    frames = dealer.recv_multipart()
    envelope = json.loads(frames[-1].decode())
    assert envelope["ok"] is True, f"Expected ok=true, got {envelope}"
    return envelope["data"]


def main():
    print("=" * 60)
    print("Real Optimizer Test (Layer 3)")
    print("=" * 60)

    print("\nDiscovering optimizer service...")
    try:
        endpoint = ServiceLocator.wait_for_service(ServiceLocator.OPTIMIZER, timeout_sec=15)
    except TimeoutError:
        print("FAIL: Optimizer service not discovered within 15s")
        print("Is optimizer.py running?")
        sys.exit(1)
    print(f"OK: Found optimizer at {endpoint.router}")

    ctx = zmq.Context()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, 35000)
    dealer.setsockopt(zmq.LINGER, 0)
    dealer.connect(endpoint.router)

    # --- Protocol round-trip ---
    print("\n--- Layer 2: Protocol Round-Trip ---")
    req = make_request(verbose=False)
    resp = send_request(dealer, req)
    assert resp["status"] == "optimal", f"Expected optimal, got {resp['status']}"
    for field in ("recommendations", "score", "binding_constraints", "solver_meta"):
        assert field in resp, f"Missing {field}"
    assert resp["solver_meta"]["solver_name"] == EXPECTED_SOLVER, \
        f"Expected solver_name={EXPECTED_SOLVER}, got {resp['solver_meta'].get('solver_name')}"
    assert resp["solver_meta"]["solver_status"] == "optimal", \
        f"Expected solver_status optimal, got {resp['solver_meta'].get('solver_status')}"
    assert resp["request_id"] == req["request_id"], "request_id not echoed"
    assert resp["request_timestamp"] == req["timestamp"], "timestamp not echoed"
    print(f"OK: optimal, solver={resp['solver_meta']['solver_name']}, "
          f"problem_size={resp['solver_meta']['problem_size']}, "
          f"wall_clock={resp['solver_meta']['wall_clock_seconds']}s")

    # --- No yield floor: nothing pre-filtered, yield_floor_pct ignored ---
    print("\n--- Layer 2: No Yield Floor (vestigial field ignored) ---")
    vresp = send_request(dealer, make_request(verbose=True))
    vrecs = vresp["recommendations"]
    n_dvs = len(req["decision_variables"])
    assert len(vrecs) == n_dvs, \
        f"Verbose should return all {n_dvs} decision variables (no pre-filter), got {len(vrecs)}"
    assert resp["solver_meta"]["problem_size"]["decision_variables"] == n_dvs, \
        "problem_size should equal full dv count (no pre-filter)"
    vids = {r["id"] for r in vrecs}
    assert "O:NVDA260515P00215000-long" in vids, \
        "Long leg must be present (no yield floor to exclude it)"
    print(f"OK: all {n_dvs} dvs are decision variables despite yield_floor_pct=0.5 (ignored)")

    # --- Bounds respected ---
    print("\n--- Layer 2: Bounds Respected ---")
    dv_by_id = {dv["id"]: dv for dv in req["decision_variables"]}
    recs = resp["recommendations"]
    for r in recs:
        assert r["target_quantity"] >= 0, f"{r['id']}: negative target"
        assert isinstance(r["target_quantity"], int), \
            f"{r['id']}: target not int ({r['target_quantity']!r})"
        dv = dv_by_id[r["id"]]
        if dv.get("closeable_only"):
            assert r["target_quantity"] <= dv["current_quantity"], \
                f"{r['id']}: closeable_only target exceeds current"
    bp_limit = (req["policy"]["bp_limit_pct"] / 100.0) * req["capital"]["net_liq"]
    used = sum(r["target_quantity"] * dv_by_id[r["id"]]["margin_per_unit"] for r in recs)
    assert used <= bp_limit + TOL, f"BP limit exceeded: {used} > {bp_limit}"
    print(f"OK: int targets, bounds + BP respected (margin ${used:,.0f}/${bp_limit:,.0f})")

    # --- Arithmetic sanity (uses verbose response: all dvs present) ---
    print("\n--- Layer 2: Arithmetic Sanity ---")
    score = vresp["score"]
    comps = score["components"]
    sums = {"theta": 0.0, "spread_cost": 0.0, "delta_ceiling": 0.0, "gamma": 0.0, "total": 0.0}
    for r in vrecs:
        oc = r["objective_contribution"]
        for k in ("theta", "spread_cost", "delta_ceiling", "gamma", "total"):
            sums[k] += oc[k]
        comp_sum = oc["theta"] + oc["spread_cost"] + oc["delta_ceiling"] + oc["gamma"]
        assert abs(oc["total"] - comp_sum) < TOL, f"{r['id']}: components don't sum to total"
    assert abs(comps["theta_total"] - sums["theta"]) < TOL, "theta_total mismatch"
    assert abs(comps["spread_cost_total"] - sums["spread_cost"]) < TOL, "spread_cost_total mismatch"
    assert abs(comps["delta_ceiling_penalty"] - sums["delta_ceiling"]) < TOL, "delta_ceiling_penalty mismatch"
    assert abs(comps["gamma_penalty"] - sums["gamma"]) < TOL, "gamma_penalty mismatch"
    assert abs(score["objective_value"] - sums["total"]) < TOL, "objective_value mismatch"
    print(f"OK: all components sum to score; objective={score['objective_value']:.4f}")

    # --- Gamma attribution: independent recomputation of the group-sum-squared ---
    print("\n--- Layer 2: Gamma Group-Sum Attribution ---")
    lam_gamma = req["policy"]["lambda_gamma"]
    group_sum = defaultdict(float)  # underlying -> Σ gamma_per_unit · target
    for r in vrecs:
        dv = dv_by_id[r["id"]]
        group_sum[r["underlying"]] += dv["gamma_per_unit"] * r["target_quantity"]
    expected_gamma_penalty = -lam_gamma * sum(s * s for s in group_sum.values())
    assert abs(comps["gamma_penalty"] - expected_gamma_penalty) < TOL, \
        (f"gamma_penalty {comps['gamma_penalty']} != independently recomputed "
         f"-λ·Σ(group_sum)² = {expected_gamma_penalty} (group-sum form broken?)")
    print(f"OK: gamma_penalty={comps['gamma_penalty']:.4f} matches -λ·Σ(Σγ·q)²={expected_gamma_penalty:.4f}")

    # --- Status: malformed_request (schema_version mismatch) ---
    print("\n--- Layer 3: malformed_request ---")
    bad = make_request()
    bad["schema_version"] = 2
    mresp = send_request(dealer, bad)
    assert mresp["status"] == "malformed_request", f"got {mresp['status']}"
    assert mresp.get("error"), "malformed_request must carry error"
    assert mresp["solver_meta"] == {"solver_name": EXPECTED_SOLVER}, \
        f"malformed_request solver_meta must be minimal, got {mresp['solver_meta']}"
    for absent in ("recommendations", "score", "binding_constraints"):
        assert absent not in mresp, f"malformed_request must omit {absent}"
    assert mresp["request_id"] == bad["request_id"], "metadata still echoed on malformed_request"
    print("OK: schema_version=2 -> malformed_request, minimal solver_meta, fields absent")

    # --- Status: infeasible (unsatisfiable delta target) ---
    print("\n--- Layer 3: infeasible ---")
    iresp = send_request(dealer, make_infeasible_request())
    assert iresp["status"] == "infeasible", f"got {iresp['status']}"
    assert iresp.get("error"), "infeasible must carry error"
    assert "in play" in iresp["error"].lower(), \
        "infeasible error should be the non-committal message (constraints 'in play')"
    bc = iresp["binding_constraints"]
    assert isinstance(bc, list) and len(bc) >= 1, "infeasible must populate binding_constraints"
    types = {c["type"] for c in bc}
    assert {"bp_limit", "delta_target"} <= types, f"expected request constraints listed, got {types}"
    for c in bc:
        assert c["value_at_solution"] is None, "infeasible value_at_solution must be null"
    for absent in ("recommendations", "score"):
        assert absent not in iresp, f"infeasible must omit {absent}"
    print(f"OK: unsatisfiable delta target -> infeasible, {len(bc)} constraints in play, recs/score absent")

    # --- Status: internal_error (NaN in a per-unit value) ---
    print("\n--- Layer 3: internal_error ---")
    nan_req = make_request()
    nan_req["decision_variables"][1]["theta_per_unit"] = float("nan")
    eresp = send_request(dealer, nan_req)
    assert eresp["status"] == "internal_error", f"got {eresp['status']}"
    assert eresp.get("error"), "internal_error must carry error"
    assert eresp["solver_meta"] == {"solver_name": EXPECTED_SOLVER}, \
        f"internal_error solver_meta must be minimal, got {eresp['solver_meta']}"
    for absent in ("recommendations", "score", "binding_constraints"):
        assert absent not in eresp, f"internal_error must omit {absent}"
    print("OK: NaN theta -> internal_error, minimal solver_meta, fields absent")

    dealer.close()
    ctx.term()

    print("\n" + "=" * 60)
    print("LAYER 3 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
