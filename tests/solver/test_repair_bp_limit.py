"""The cardinality cap must not turn a hedged book into a BP breach.

`solver_request_repair_bp.json` is a WWW strangle — short put (+45 delta) and short call
(-45 delta), both held 10 and close-only — with max_positions_per_underlying = 1 and
bp_limit = 11,000. The continuous optimum holds both legs at the BP limit with net delta 0.
The cardinality cap keeps one leg and drops the other, throwing WWW net delta to +450 shares
(+45% of net liq, far outside the ±5% band).

Before the fix the stock repair then shorted 400 shares to drag the band back — adding 20,000
of margin with no BP check, a 25,500 book over the 11,000 limit (the run-87e615b0 mechanism).
Now the cap drop triggers a re-solve with the dropped contract fixed: it is held and
close-only, so it is frozen at its current quantity (we keep the holding, we do not
force-close it), and the re-solve re-hedges within BP and the band. The book stays within
bp_limit, the hedge is preserved, and the stock repair is left with only rounding error.
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
    """Every stage — solve, netting, rounding, cap, resolve, repair — stays within bp_limit.
    The cap and the stock repair are the steps that used to break it."""
    tr = optimizer.pipeline_trace(_request())
    for step in tr["steps"]:
        assert step["bp_used"] <= tr["bp_limit"] + 1e-6, \
            f"{step['step']} uses {step['bp_used']:.2f} > bp_limit {tr['bp_limit']:.2f}"


def test_cardinality_drop_preserves_the_held_hedge_feasibly() -> None:
    """The dropped leg is held and close-only: the re-solve freezes it at its current
    quantity rather than closing it, so both legs remain and the book is feasible — not a
    fabricated infeasibility, not a BP breach."""
    resp = optimizer.handle_optimize(copy.deepcopy(_request()))
    assert resp["status"] == optimizer.STATUS_OPTIMAL, resp.get("error")
    assert not resp["binding_constraints"], resp["binding_constraints"]

    recs = {r["id"]: r for r in resp["recommendations"]}
    req = _request()
    held_close = [dv for dv in req["decision_variables"]
                  if dv["type"] == "option" and dv.get("closeable_only") and dv["current_quantity"] > 0]
    kept = [dv for dv in held_close if recs[dv["id"]]["optimal_quantity"] > 0]
    assert len(kept) == len(held_close), \
        "a held close-only hedge leg was closed by the cardinality cap instead of frozen"

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
