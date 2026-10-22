"""Regression: the stock repair must respect bp_limit.

`solver_request_repair_bp.json` is a WWW strangle — short put (+45 delta) and short call
(-45 delta), both held 10 — with max_positions_per_underlying = 1 and bp_limit = 11,000.
The continuous optimum holds both legs at the BP limit with net delta 0. The cardinality
cap drops one leg, throwing WWW net delta to +450 shares (+45% of net liq, far outside the
±5% band). The stock repair hedges that back toward the band.

Before the fix the repair shorted 400 shares (+20,000 margin → a 25,500 book, over the
11,000 limit): a cap-dropped hedge, rebuilt in stock with no BP check. Now the repair moves
only as many shares as the remaining BP allows (110, to 11,000) and the re-check reports the
residual delta-band breach rather than a fabricated BP breach. This mirrors run 87e615b0.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import optimizer  # noqa: E402

_REQ = Path(__file__).parent / "solver_request_repair_bp.json"


def _request() -> dict:
    return json.loads(_REQ.read_text(encoding="utf-8"))


def test_no_post_solve_step_exceeds_bp_limit() -> None:
    """Every stage — solve, netting, rounding, cap, repair — stays within bp_limit.
    The repair is the only step that can raise margin; this is what used to break."""
    tr = optimizer.pipeline_trace(_request())
    bp_limit = tr["bp_limit"]
    for step in tr["steps"]:
        assert step["bp_used"] <= bp_limit + 1e-6, \
            f"{step['step']} uses {step['bp_used']:.2f} > bp_limit {bp_limit:.2f}"


def test_repair_reports_residual_band_breach_not_a_bp_breach() -> None:
    resp = optimizer.handle_optimize(copy.deepcopy(_request()))
    assert resp["status"] == optimizer.STATUS_INFEASIBLE
    types = {b["type"] for b in resp["binding_constraints"]}
    assert optimizer.BCT_DELTA_TARGET in types, \
        f"expected the residual delta-band breach, got {resp['binding_constraints']}"
    assert optimizer.BCT_BP_LIMIT not in types, \
        f"BP is respected now; it must not appear as a breach: {resp['binding_constraints']}"
