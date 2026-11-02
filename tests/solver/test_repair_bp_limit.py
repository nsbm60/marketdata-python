"""No post-solve step exceeds the BP limit, and the held hedge is returned intact.

`solver_request_repair_bp.json` is a WWW strangle — short put (+45 delta) and short call
(-45 delta), both held 10 and close-only — with bp_limit = 11,000. The continuous optimum
holds both legs at the BP limit with net delta 0. With the position cap removed (a leftover
from the raw-theta objective), nothing drops a hedge leg: the solver returns both legs, the
book stays within bp_limit through every post-solve step, and the stock repair is left with
only rounding error. (max_positions_per_underlying is still in the fixture and is ignored.)
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import optimizer  # noqa: E402

_REQ = Path(__file__).parent / "solver_request_repair_bp.json"


def _request() -> dict:
    return json.loads(_REQ.read_text(encoding="utf-8"))


def test_no_post_solve_step_exceeds_bp_limit() -> None:
    """Every stage — solve, netting, rounding, repair — stays within bp_limit. The stock
    repair is the step that used to break it."""
    tr = optimizer.pipeline_trace(_request())
    for step in tr["steps"]:
        assert step["bp_used"] <= tr["bp_limit"] + 1e-6, \
            f"{step['step']} uses {step['bp_used']:.2f} > bp_limit {tr['bp_limit']:.2f}"


def test_held_balanced_hedge_is_returned_intact_and_feasible() -> None:
    """Both legs are held and close-only; with no position cap to drop one, the continuous
    solve holds both at net delta 0, and the book is feasible — within BP and the band."""
    resp = optimizer.handle_optimize(copy.deepcopy(_request()))
    assert resp["status"] == optimizer.STATUS_OPTIMAL, resp.get("error")
    assert not resp["binding_constraints"], resp["binding_constraints"]

    recs = {r["id"]: r for r in resp["recommendations"]}
    req = _request()
    held_close = [dv for dv in req["decision_variables"]
                  if dv["type"] == "option" and dv.get("closeable_only") and dv["current_quantity"] > 0]
    kept = [dv for dv in held_close if recs[dv["id"]]["optimal_quantity"] > 0]
    assert len(kept) == len(held_close), \
        "a held close-only hedge leg was not returned"

    # Within BP and inside the band, computed from the recommended book.
    net_liq = req["capital"]["net_liq"]
    bp_limit = req["policy"]["bp_limit_pct"] / 100.0 * net_liq
    dvs = {dv["id"]: dv for dv in req["decision_variables"]}
    used = sum(r["optimal_quantity"] * dvs[r["id"]]["margin_per_unit"] for r in resp["recommendations"])
    assert used <= bp_limit + 1e-6, f"BP {used:.2f} > limit {bp_limit:.2f}"
    net = {}
    for r in resp["recommendations"]:
        net[r["underlying"]] = net.get(r["underlying"], 0.0) + dvs[r["id"]]["delta_per_unit"] * r["optimal_quantity"]
    for pu in req["per_underlying"]:
        pct = pu["spot"] * net.get(pu["underlying"], 0.0) / net_liq * 100.0
        assert abs(pct - pu["delta_target_pct"]) <= pu["delta_tolerance_pct"] + 1e-6, \
            f"{pu['underlying']} net delta {pct:.4f}% outside band"
