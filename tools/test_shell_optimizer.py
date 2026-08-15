#!/usr/bin/env python3
"""
Test harness for the shell optimizer service.
Covers acceptance criteria layers 3-6.

Start the shell optimizer in one terminal:
    python shell_optimizer.py

Run this test in another:
    python tools/test_shell_optimizer.py
"""

import json
import sys
import uuid

sys.path.insert(0, ".")

import zmq
from discovery.service_locator import ServiceLocator

# ─────────────────────────────────────────────────────────────
# Synthetic request data
# ─────────────────────────────────────────────────────────────

def make_request(verbose=False, num_dvs=20):
    """Build a synthetic optimize request with num_dvs decision variables."""
    dvs = []
    for i in range(num_dvs):
        underlying = ["NVDA", "MU", "LITE", "LYB"][i % 4]
        direction = "short" if i % 2 == 0 else "long"
        strike = 200 + i * 5
        # Short options have positive theta; long options negative
        theta = (15.0 - i * 0.5) if direction == "short" else -(5.0 + i * 0.3)
        dvs.append({
            "id": f"O:{underlying}260515P00{strike * 1000:08.0f}-{direction}",
            "underlying": underlying,
            "type": "option",
            "direction": direction,
            "current_quantity": 2 if i < 3 else 0,
            "delta_per_unit": 25.0 if direction == "short" else -25.0,
            "theta_per_unit": theta,
            "gamma_per_unit": -1.5 if direction == "short" else 1.5,
            "margin_per_unit": 4000.0,
            "premium_per_unit": 250.0 if direction == "short" else -250.0,
            "spread_cost_per_unit": 8.0,
            "iv": 0.35,
            "days_to_expiry": 5,
            "strike": float(strike),
            "expiry": "2026-05-15",
        })

    return {
        "schema_version": 1,
        "request_id": f"test-{uuid.uuid4().hex[:8]}",
        "policy_version": "v1",
        "timestamp": "2026-05-13T12:00:00Z",
        "verbose": verbose,
        "decision_variables": dvs,
        "per_underlying": [
            {"underlying": "NVDA", "delta_target": 0, "tolerance": 25, "max_positions": 8, "tradeable": True},
            {"underlying": "MU", "delta_target": 0, "tolerance": 25, "max_positions": 8, "tradeable": True},
            {"underlying": "LITE", "delta_target": 0, "tolerance": 25, "max_positions": 8, "tradeable": True},
            {"underlying": "LYB", "delta_target": 0, "tolerance": 25, "max_positions": 8, "tradeable": True},
        ],
        "policy": {
            "bp_limit_pct": 60,
            "max_positions_per_underlying": 8,
            "concentration_limit_pct": 30,
            "yield_floor_pct": 0.1,
            "delta_ceiling_threshold": 0.30,
            "lambda_delta_ceiling": 1.0,
            "lambda_gamma": 1.0,
            "lambda_spread": 0.65,
            "default_delta_tolerance": 25.0,
        },
        "capital": {
            "net_liq": 250000.0,
            "current_bp_usage": 45000.0,
            "current_margin": 32000.0,
        },
    }


def send_request(dealer, request):
    """Send an optimize request, return parsed response data."""
    dealer.send_multipart([b"", json.dumps({"op": "optimize", **request}).encode()])
    frames = dealer.recv_multipart()
    envelope = json.loads(frames[-1].decode())
    assert envelope["ok"] is True, f"Expected ok=true, got {envelope}"
    return envelope["data"]


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Shell Optimizer Test")
    print("=" * 60)

    # Discover the optimizer
    print("\nDiscovering optimizer service...")
    try:
        endpoint = ServiceLocator.wait_for_service(ServiceLocator.OPTIMIZER, timeout_sec=15)
    except TimeoutError:
        print("FAIL: Optimizer service not discovered within 15s")
        print("Is shell_optimizer.py running?")
        sys.exit(1)
    print(f"OK: Found optimizer at {endpoint.router}")

    ctx = zmq.Context()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, 10000)
    dealer.setsockopt(zmq.LINGER, 0)
    dealer.connect(endpoint.router)

    # --- Layer 3: Optimal response ---
    print("\n--- Layer 3: Optimal Response ---")
    req = make_request(verbose=False, num_dvs=20)
    resp = send_request(dealer, req)

    assert resp["status"] == "optimal", f"Expected optimal, got {resp['status']}"
    assert "recommendations" in resp, "Missing recommendations"
    assert "score" in resp, "Missing score"
    assert "binding_constraints" in resp, "Missing binding_constraints"
    assert resp["binding_constraints"] == [], f"Expected empty binding_constraints, got {resp['binding_constraints']}"
    assert "solver_meta" in resp, "Missing solver_meta"
    assert resp["solver_meta"]["solver_name"] == "shell", f"Expected shell solver, got {resp['solver_meta']}"
    assert resp["request_id"] == req["request_id"], "request_id not echoed"
    assert resp["policy_version"] == req["policy_version"], "policy_version not echoed"
    assert resp["request_timestamp"] == req["timestamp"], "timestamp not echoed"
    assert "response_timestamp" in resp, "Missing response_timestamp"

    recs = resp["recommendations"]
    # Should have selected 5 (default) + 3 with current_quantity > 0 (some may overlap)
    selected = [r for r in recs if r["target_quantity"] > 0]
    held = [r for r in recs if r["current_quantity"] > 0]
    assert len(selected) <= 5, f"Expected at most 5 selected, got {len(selected)}"
    assert len(held) >= 1, f"Expected at least 1 held position in recommendations"

    # Verify score arithmetic
    score = resp["score"]
    theta_sum = sum(r["objective_contribution"]["theta"] for r in recs)
    spread_sum = sum(r["objective_contribution"]["spread_cost"] for r in recs)
    assert abs(score["components"]["theta_total"] - theta_sum) < 0.01, "theta_total mismatch"
    assert abs(score["components"]["spread_cost_total"] - spread_sum) < 0.01, "spread_cost_total mismatch"
    expected_obj = theta_sum + spread_sum
    assert abs(score["objective_value"] - expected_obj) < 0.01, "objective_value mismatch"

    # Verify diagnostic fields echoed
    for r in recs:
        if r["type"] == "option":
            assert "strike" in r, f"Missing strike on option rec {r['id']}"
            assert "expiry" in r, f"Missing expiry on option rec {r['id']}"

    print(f"OK: {len(recs)} recommendations, {len(selected)} selected, objective={score['objective_value']:.2f}")

    # --- Layer 4: Verbose flag ---
    print("\n--- Layer 4: Verbose Flag ---")

    # Non-verbose (already tested above)
    non_verbose_count = len(recs)

    # Verbose
    req_v = make_request(verbose=True, num_dvs=20)
    resp_v = send_request(dealer, req_v)
    verbose_recs = resp_v["recommendations"]
    assert len(verbose_recs) == 20, f"Verbose: expected 20 recs (all dvs), got {len(verbose_recs)}"
    assert len(verbose_recs) > non_verbose_count, f"Verbose should return more recs than non-verbose"

    # Unselected variables should have zero contribution
    unselected = [r for r in verbose_recs if r["target_quantity"] == 0 and r["current_quantity"] == 0]
    for r in unselected:
        assert r["objective_contribution"]["total"] == 0, f"Unselected {r['id']} should have zero contribution"

    print(f"OK: Non-verbose={non_verbose_count} recs, verbose={len(verbose_recs)} recs")

    # --- Layer 5: Determinism ---
    print("\n--- Layer 5: Determinism ---")
    req1 = make_request(verbose=True, num_dvs=15)
    req2 = {**req1, "request_id": req1["request_id"]}  # same request
    resp1 = send_request(dealer, req1)
    resp2 = send_request(dealer, req2)

    # Compare everything except response_timestamp
    resp1.pop("response_timestamp")
    resp2.pop("response_timestamp")
    # wall_clock_seconds will differ slightly
    resp1["solver_meta"].pop("wall_clock_seconds", None)
    resp2["solver_meta"].pop("wall_clock_seconds", None)

    assert resp1 == resp2, f"Determinism failed: responses differ"
    print("OK: Identical requests produce identical responses")

    # --- Layer 6: Missing op / unknown op (service library behavior) ---
    print("\n--- Layer 6: Service Library Error Handling ---")
    dealer.send_multipart([b"", json.dumps({"op": "nonexistent"}).encode()])
    frames = dealer.recv_multipart()
    err = json.loads(frames[-1].decode())
    assert err["ok"] is False, "Expected error for unknown op"
    assert "unsupported op" in err["error"], f"Expected unsupported op error, got {err['error']}"
    print(f"OK: Unknown op handled — {err['error']}")

    dealer.close()
    ctx.term()

    print("\n" + "=" * 60)
    print("ALL LAYERS PASSED")
    print("=" * 60)
    print("\nNote: Status injection (Layer 5 from spec) requires restarting")
    print("the shell with different SHELL_OPTIMIZER_FORCE_STATUS values.")
    print("Test manually or with a wrapper script.")


if __name__ == "__main__":
    main()
