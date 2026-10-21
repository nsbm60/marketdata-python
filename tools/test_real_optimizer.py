#!/usr/bin/env python3
"""
Integration harness for the real optimizer service (optimizer.py).

The continuous guard rail for the integration-first build per
pc-solver-spec-draft.md, extended for the run-loop amendments
(optimizer-run-loop-spec-draft.md §3, 2026-10-21):

  - execution cost is charged on |target − current| (holding is free);
  - interim holding horizon h = min(days_to_expiry, horizon_days), with
    fractional days_to_expiry: theta × h, gamma scaled by √h in the group sum;
  - min_improvement and action hold|trade, never hold an invalid book.

Start the real optimizer in one terminal:
    python optimizer.py

Run this test in another:
    python tools/test_real_optimizer.py

Scope: protocol round-trip, arithmetic sanity for the full formulation
(penalties, hard constraints, post-solve cardinality), the status taxonomy,
and the run-loop behaviour above. The gamma penalty is independently
recomputed from the response to verify the per-underlying √h group-sum
attribution.
"""

import math
import sys
import uuid
from collections import defaultdict

sys.path.insert(0, ".")

from control_client import ControlSocket
from discovery import ControlService, ServiceLocator

EXPECTED_SOLVER = "clarabel"  # default backend; override if OPTIMIZER_SOLVER is set
TOL = 1e-4  # arithmetic-sanity tolerance
HORIZON_DAYS = 5.0

POLICY = {
    "bp_limit_pct": 60, "max_positions_per_underlying": 8, "concentration_limit_pct": 30,
    "delta_ceiling_threshold": 0.30, "lambda_delta_ceiling": 1.0, "lambda_gamma": 1.0,
    "lambda_spread": 1.0, "min_improvement": 0.0, "default_delta_tolerance": 25.0,
}


def make_request(verbose=False):
    """Synthetic request: multiple underlyings, held + candidate + closeable_only,
    short (theta>0) and long (theta<0) directions, an equity leg, spot per
    underlying, fractional days_to_expiry (including a same-day expiry at 0.25).
    Margins realistic so BP can bind."""
    dvs = [
        {"id": "O:NVDA260515P00215000-short", "underlying": "NVDA", "type": "option",
         "direction": "short", "current_quantity": 2, "delta_per_unit": 27.5,
         "theta_per_unit": 12.5, "gamma_per_unit": -1.8, "margin_per_unit": 4327.0,
         "premium_per_unit": 280.0, "exec_cost_per_unit": 4.0, "iv": 0.319,
         "days_to_expiry": 3.235, "strike": 215.0, "expiry": "2026-05-15"},
        {"id": "O:NVDA260515P00220000-short", "underlying": "NVDA", "type": "option",
         "direction": "short", "current_quantity": 0, "delta_per_unit": 35.0,
         "theta_per_unit": 15.2, "gamma_per_unit": -2.1, "margin_per_unit": 4527.0,
         "premium_per_unit": 320.0, "exec_cost_per_unit": 4.5, "iv": 0.325,
         "days_to_expiry": 3.235, "strike": 220.0, "expiry": "2026-05-15"},
        {"id": "O:NVDA260515P00215000-long", "underlying": "NVDA", "type": "option",
         "direction": "long", "current_quantity": 0, "delta_per_unit": -27.5,
         "theta_per_unit": -12.5, "gamma_per_unit": 1.8, "margin_per_unit": 280.0,
         "premium_per_unit": -280.0, "exec_cost_per_unit": 4.0, "iv": 0.319,
         "days_to_expiry": 3.235, "strike": 215.0, "expiry": "2026-05-15"},
        {"id": "O:MU260512C00120000-short", "underlying": "MU", "type": "option",
         "direction": "short", "current_quantity": 0, "delta_per_unit": -22.0,
         "theta_per_unit": 30.0, "gamma_per_unit": -3.0, "margin_per_unit": 2200.0,
         "premium_per_unit": 60.0, "exec_cost_per_unit": 3.0, "iv": 0.41,
         "days_to_expiry": 0.25, "strike": 120.0, "expiry": "2026-05-12"},
        {"id": "O:MU260529C00125000-short", "underlying": "MU", "type": "option",
         "direction": "short", "current_quantity": 0, "delta_per_unit": -20.0,
         "theta_per_unit": 6.1, "gamma_per_unit": -0.6, "margin_per_unit": 2400.0,
         "premium_per_unit": 410.0, "exec_cost_per_unit": 5.0, "iv": 0.40,
         "days_to_expiry": 17.235, "strike": 125.0, "expiry": "2026-05-29"},
        {"id": "O:TSLA260515P00250000-short", "underlying": "TSLA", "type": "option",
         "direction": "short", "current_quantity": 2, "closeable_only": True,
         "delta_per_unit": 18.0, "theta_per_unit": 8.4, "gamma_per_unit": -1.2,
         "margin_per_unit": 4980.0, "premium_per_unit": 175.0, "exec_cost_per_unit": 5.5,
         "iv": 0.412, "days_to_expiry": 3.235, "strike": 250.0, "expiry": "2026-05-15"},
        {"id": "NVDA-equity-long", "underlying": "NVDA", "type": "equity",
         "direction": "long", "current_quantity": 100, "delta_per_unit": 1.0,
         "theta_per_unit": 0, "gamma_per_unit": 0, "margin_per_unit": 21.65,
         "premium_per_unit": 216.49, "exec_cost_per_unit": 0.005},
    ]
    return {
        "schema_version": 1,
        "request_id": f"test-{uuid.uuid4().hex[:8]}",
        "policy_version": "v3",
        "timestamp": "2026-05-12T14:22:37Z",
        "horizon_days": HORIZON_DAYS,
        "verbose": verbose,
        "decision_variables": dvs,
        "per_underlying": [
            {"underlying": "NVDA", "spot": 215.0, "delta_target": 0, "tolerance": 25, "max_positions": 8, "tradeable": True},
            {"underlying": "MU", "spot": 118.0, "delta_target": 0, "tolerance": 25, "max_positions": 8, "tradeable": True},
            {"underlying": "TSLA", "spot": 250.0, "delta_target": 0, "tolerance": 25, "max_positions": 8, "tradeable": False},
        ],
        "policy": dict(POLICY),
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
        "timestamp": "2026-05-12T14:22:37Z",
        "horizon_days": HORIZON_DAYS,
        "verbose": False,
        "decision_variables": [{
            "id": "O:ZZZ260515P00100000-short", "underlying": "ZZZ", "type": "option",
            "direction": "short", "current_quantity": 0, "delta_per_unit": 30.0,
            "theta_per_unit": 10.0, "gamma_per_unit": -1.0, "margin_per_unit": 1000.0,
            "premium_per_unit": 100.0, "exec_cost_per_unit": 2.5, "iv": 0.3,
            "days_to_expiry": 3.235, "strike": 100.0, "expiry": "2026-05-15"}],
        "per_underlying": [
            {"underlying": "ZZZ", "spot": 100.0, "delta_target": -100000, "tolerance": 1,
             "max_positions": 8, "tradeable": True}],
        "policy": dict(POLICY),
        "capital": {"net_liq": 250000.0, "current_bp_usage": 0.0, "current_margin": 0.0},
    }


def make_switch_request(exec_cost, min_improvement=0.0, held_qty=2):
    """One underlying, two near-identical short puts. A is held; B has slightly
    more theta (+0.5/day). The delta band (tolerance 40, delta 20 each) caps the
    book at 2 contracts, so the only question is whether switching A→B pays.
    Gamma is 0 so the comparison is theta over the horizon vs execution cost."""
    def leg(name, theta, current):
        return {"id": f"O:ZZZ260515P00{name}-short", "underlying": "ZZZ", "type": "option",
                "direction": "short", "current_quantity": current, "delta_per_unit": 20.0,
                "theta_per_unit": theta, "gamma_per_unit": 0.0, "margin_per_unit": 1000.0,
                "premium_per_unit": 150.0, "exec_cost_per_unit": exec_cost, "iv": 0.3,
                "days_to_expiry": 3.235, "strike": 100.0, "expiry": "2026-05-15"}
    policy = dict(POLICY)
    policy["min_improvement"] = min_improvement
    return {
        "schema_version": 1,
        "request_id": f"test-sw-{uuid.uuid4().hex[:8]}",
        "policy_version": "v3",
        "timestamp": "2026-05-12T14:22:37Z",
        "horizon_days": HORIZON_DAYS,
        "verbose": True,
        "decision_variables": [leg("100000A", 10.0, held_qty), leg("100000B", 10.5, 0)],
        "per_underlying": [
            {"underlying": "ZZZ", "spot": 100.0, "delta_target": 0, "tolerance": 40,
             "max_positions": 8, "tradeable": True}],
        "policy": policy,
        "capital": {"net_liq": 250000.0, "current_bp_usage": 0.0, "current_margin": 0.0},
    }


def horizon(dv):
    if dv["type"] == "option":
        return min(dv["days_to_expiry"], HORIZON_DAYS)
    return HORIZON_DAYS


def send_request(dealer, request):
    envelope = dealer.request({"op": "optimize", **request})
    assert envelope["ok"] is True, f"Expected ok=true, got {envelope}"
    return envelope["data"]


def by_id(resp):
    return {r["id"]: r for r in resp["recommendations"]}


def check_arithmetic(req, resp, label):
    """Layer 2 sums plus the exact per-term formulas, for any verbose response."""
    dv_by_id = {dv["id"]: dv for dv in req["decision_variables"]}
    lam = req["policy"]
    score, comps = resp["score"], resp["score"]["components"]
    sums = defaultdict(float)
    group_x = defaultdict(float)
    for r in resp["recommendations"]:
        dv = dv_by_id[r["id"]]
        oc = r["objective_contribution"]
        h = horizon(dv)
        t, c = r["target_quantity"], r["current_quantity"]
        for k in ("theta", "exec_cost", "delta_ceiling", "gamma", "total"):
            sums[k] += oc[k]
        assert abs(oc["total"] - (oc["theta"] + oc["exec_cost"] + oc["delta_ceiling"] + oc["gamma"])) < TOL, \
            f"{label} {r['id']}: components don't sum to total"
        assert abs(oc["theta"] - dv["theta_per_unit"] * h * t) < TOL, \
            f"{label} {r['id']}: theta {oc['theta']} != theta·h·target ({h=})"
        assert abs(oc["exec_cost"] - (-lam["lambda_spread"] * dv["exec_cost_per_unit"] * abs(t - c))) < TOL, \
            f"{label} {r['id']}: exec_cost not −λ·exec·|target−current|"
        group_x[r["underlying"]] += math.sqrt(h) * dv["gamma_per_unit"] * t
    assert abs(comps["theta_total"] - sums["theta"]) < TOL, f"{label}: theta_total mismatch"
    assert abs(comps["exec_cost_total"] - sums["exec_cost"]) < TOL, f"{label}: exec_cost_total mismatch"
    assert abs(comps["delta_ceiling_penalty"] - sums["delta_ceiling"]) < TOL, f"{label}: delta_ceiling_penalty mismatch"
    assert abs(comps["gamma_penalty"] - sums["gamma"]) < TOL, f"{label}: gamma_penalty mismatch"
    assert abs(score["objective_value"] - sums["total"]) < TOL, f"{label}: objective_value mismatch"
    expected_gamma = -lam["lambda_gamma"] * sum(s * s for s in group_x.values())
    assert abs(comps["gamma_penalty"] - expected_gamma) < TOL, \
        f"{label}: gamma_penalty {comps['gamma_penalty']} != −λ·Σ(Σ√h·γ·q)² = {expected_gamma}"


def main():
    print("=" * 60)
    print("Real Optimizer Test (Layer 3 + run-loop amendments)")
    print("=" * 60)

    print("\nDiscovering optimizer service...")
    try:
        endpoint = ServiceLocator.control(ControlService.OPTIMIZER, timeout_sec=15)
    except TimeoutError:
        print("FAIL: Optimizer service not discovered within 15s")
        print("Is optimizer.py running?")
        sys.exit(1)
    print(f"OK: Found optimizer at {endpoint.router}")

    dealer = ControlSocket(endpoint.router, timeout_ms=35000)

    # --- Protocol round-trip ---
    print("\n--- Layer 2: Protocol Round-Trip ---")
    req = make_request(verbose=False)
    resp = send_request(dealer, req)
    assert resp["status"] == "optimal", f"Expected optimal, got {resp['status']}"
    for field in ("action", "recommendations", "score", "binding_constraints", "solver_meta"):
        assert field in resp, f"Missing {field}"
    assert resp["action"] in ("trade", "hold"), f"bad action {resp['action']!r}"
    for field in ("optimal_objective_value", "current_objective_value"):
        assert field in resp["score"], f"score missing {field}"
    assert resp["solver_meta"]["solver_name"] == EXPECTED_SOLVER, \
        f"Expected solver_name={EXPECTED_SOLVER}, got {resp['solver_meta'].get('solver_name')}"
    assert resp["solver_meta"]["solver_status"] == "optimal", \
        f"Expected solver_status optimal, got {resp['solver_meta'].get('solver_status')}"
    assert resp["request_id"] == req["request_id"], "request_id not echoed"
    assert resp["request_timestamp"] == req["timestamp"], "timestamp not echoed"
    for r in resp["recommendations"]:
        assert "optimal_quantity" in r, f"{r['id']}: optimal_quantity missing"
    print(f"OK: optimal, action={resp['action']}, solver={resp['solver_meta']['solver_name']}, "
          f"problem_size={resp['solver_meta']['problem_size']}, "
          f"wall_clock={resp['solver_meta']['wall_clock_seconds']}s")

    # --- Verbose scope: every decision variable comes back, none pre-filtered ---
    print("\n--- Layer 2: Verbose Scope ---")
    vreq = make_request(verbose=True)
    vresp = send_request(dealer, vreq)
    vrecs = vresp["recommendations"]
    n_dvs = len(vreq["decision_variables"])
    assert len(vrecs) == n_dvs, f"Verbose should return all {n_dvs} decision variables, got {len(vrecs)}"
    assert vresp["solver_meta"]["problem_size"]["decision_variables"] == n_dvs, \
        "problem_size should equal full dv count"
    print(f"OK: all {n_dvs} decision variables returned")

    # --- Bounds respected ---
    print("\n--- Layer 2: Bounds Respected ---")
    dv_by_id = {dv["id"]: dv for dv in vreq["decision_variables"]}
    for r in vrecs:
        assert r["target_quantity"] >= 0, f"{r['id']}: negative target"
        assert isinstance(r["optimal_quantity"], int), \
            f"{r['id']}: optimal not int ({r['optimal_quantity']!r})"
        if dv_by_id[r["id"]].get("closeable_only"):
            assert r["optimal_quantity"] <= dv_by_id[r["id"]]["current_quantity"], \
                f"{r['id']}: closeable_only target exceeds current"
    bp_limit = (vreq["policy"]["bp_limit_pct"] / 100.0) * vreq["capital"]["net_liq"]
    used = sum(r["optimal_quantity"] * dv_by_id[r["id"]]["margin_per_unit"] for r in vrecs)
    assert used <= bp_limit + TOL, f"BP limit exceeded: {used} > {bp_limit}"
    print(f"OK: int targets, bounds + BP respected (margin ${used:,.0f}/${bp_limit:,.0f})")

    # --- Arithmetic sanity, horizon, exec cost, √h gamma attribution ---
    print("\n--- Layer 2: Arithmetic, Horizon, Exec Cost, Gamma ---")
    check_arithmetic(vreq, vresp, "main")
    same_day = by_id(vresp)["O:MU260512C00120000-short"]
    print(f"OK: components sum; theta = θ·min(dte,5)·q (same-day dte 0.25 -> h=0.25, "
          f"target={same_day['target_quantity']}); exec = −λ·exec·|Δq|; gamma = −λ·Σ(Σ√h·γ·q)²")

    # --- Holding is free ---
    print("\n--- Run loop: Holding Is Free ---")
    for r in vrecs:
        if r["target_quantity"] == r["current_quantity"]:
            assert abs(r["objective_contribution"]["exec_cost"]) < TOL, \
                f"{r['id']}: exec_cost charged on an unchanged position"
    print("OK: unchanged positions pay no execution cost")

    # --- Incumbency: switching must pay for itself ---
    print("\n--- Run loop: Incumbency ---")
    # Gain from switching 2 contracts A→B: 0.5 × 3.235 × 2 = 3.235. Cost: 2 legs × 2 × exec.
    sreq = make_switch_request(exec_cost=50.0)
    sticky = send_request(dealer, sreq)
    a = by_id(sticky)["O:ZZZ260515P00100000A-short"]["target_quantity"]
    b = by_id(sticky)["O:ZZZ260515P00100000B-short"]["target_quantity"]
    assert (a, b) == (2, 0), f"expensive switch should keep A: A={a}, B={b}"
    check_arithmetic(sreq, sticky, "sticky")
    cheap = send_request(dealer, make_switch_request(exec_cost=0.01))
    assert by_id(cheap)["O:ZZZ260515P00100000A-short"]["target_quantity"] == 0, "cheap switch should close A"
    assert by_id(cheap)["O:ZZZ260515P00100000B-short"]["target_quantity"] == 2, "cheap switch should open B"
    assert cheap["action"] == "trade", f"cheap switch action {cheap['action']}"
    print("OK: exec 50 -> keeps A (switch doesn't pay); exec 0.01 -> switches to B, action=trade")

    # --- Hold: gain below min_improvement, valid current book ---
    print("\n--- Run loop: Hold ---")
    hreq = make_switch_request(exec_cost=0.01, min_improvement=1000.0)
    hresp = send_request(dealer, hreq)
    assert hresp["status"] == "optimal", f"hold keeps status optimal, got {hresp['status']}"
    assert hresp["action"] == "hold", f"expected hold, got {hresp['action']}"
    for r in hresp["recommendations"]:
        assert r["target_quantity"] == r["current_quantity"], f"{r['id']}: hold must recommend no trade"
    assert by_id(hresp)["O:ZZZ260515P00100000B-short"]["optimal_quantity"] == 2, \
        "optimal_quantity must still record the solver's book on hold"
    s = hresp["score"]
    assert abs(s["objective_value"] - s["current_objective_value"]) < TOL, "hold score is the current book"
    assert s["optimal_objective_value"] > s["current_objective_value"], "gap should be positive here"
    check_arithmetic(hreq, hresp, "hold")
    print(f"OK: action=hold, targets=current, optimal_quantity kept; "
          f"gap={s['optimal_objective_value'] - s['current_objective_value']:.4f} < 1000")

    # --- Never hold a book that breaks a hard constraint ---
    print("\n--- Run loop: No Hold On An Invalid Book ---")
    bad_book = make_switch_request(exec_cost=0.01, min_improvement=1e9, held_qty=5)  # delta 100 > band 40
    bresp = send_request(dealer, bad_book)
    assert bresp["action"] == "trade", f"out-of-band current book must not hold, got {bresp['action']}"
    total_target = sum(r["target_quantity"] for r in bresp["recommendations"])
    assert total_target <= 2, f"recommended book should be back in band, got {total_target} contracts"
    print("OK: current book out of band -> action=trade despite min_improvement=1e9")

    # --- Option without days_to_expiry / negative days_to_expiry -> ok:false ---
    print("\n--- Inputs: days_to_expiry Required For Options ---")
    for mutate, label in ((lambda dv: dv.pop("days_to_expiry"), "missing"),
                          (lambda dv: dv.__setitem__("days_to_expiry", -0.1), "negative")):
        r = make_request()
        mutate(r["decision_variables"][0])
        env = dealer.request({"op": "optimize", **r})
        assert env["ok"] is False, f"{label} days_to_expiry should be rejected, got {env}"
        print(f"OK: {label} days_to_expiry -> ok:false ({env.get('error')})")

    # --- Status: malformed_request (schema_version mismatch) ---
    print("\n--- Layer 3: malformed_request ---")
    bad = make_request()
    bad["schema_version"] = 2
    mresp = send_request(dealer, bad)
    assert mresp["status"] == "malformed_request", f"got {mresp['status']}"
    assert mresp.get("error"), "malformed_request must carry error"
    assert mresp["solver_meta"] == {"solver_name": EXPECTED_SOLVER}, \
        f"malformed_request solver_meta must be minimal, got {mresp['solver_meta']}"
    for absent in ("recommendations", "score", "binding_constraints", "action"):
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
    for absent in ("recommendations", "score", "action"):
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
    for absent in ("recommendations", "score", "binding_constraints", "action"):
        assert absent not in eresp, f"internal_error must omit {absent}"
    print("OK: NaN theta -> internal_error, minimal solver_meta, fields absent")

    dealer.close()

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
