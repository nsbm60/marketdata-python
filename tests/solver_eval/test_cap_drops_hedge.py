"""Eval step 2: the cap_drops_hedge case proves the solver's cardinality/repair fixes.

Market hand_built_cap_drops_hedge is the 87e615b0 mechanism in miniature — a held balanced
short strangle with max_positions=1, so the per-underlying cap must drop one hedge leg. The
three fixes under test:
  1. equity is excluded from the per-underlying cap (it is the hedge, not a position);
  2. the re-solve after the cap freezes the dropped held leg and re-hedges within BP/band;
  3. the stock repair stays within BP (1e7a077), left with only rounding error.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parents[1]))
import optimizer  # noqa: E402
import tools.solver_eval as se  # noqa: E402


def _case() -> dict:
    return json.loads((_HERE / "cases" / "cap_drops_hedge.json").read_text(encoding="utf-8"))


def _request() -> dict:
    return se._market_request("hand_built_cap_drops_hedge")


def test_case_meets_its_expectations() -> None:
    m = se.evaluate_case(_case())
    assert m["status"] != "infeasible"
    assert all(ok for _n, ok, _d in m["checks"]), [c for c in m["checks"] if not c[1]]


def test_cap_drops_a_hedge_then_the_resolve_recovers_within_bp_and_band() -> None:
    """The cap breaks the book (a dropped leg swings net delta far out of band); the re-solve
    brings it back within BP and the band — not the unbounded stock repair that used to."""
    tr = optimizer.pipeline_trace(_request())
    steps = {s["step"]: s for s in tr["steps"]}
    assert "resolve" in steps, "cap did not bite — the case no longer exercises the fix"
    # The cap step is out of band (it dropped a hedge leg).
    cap_pct = max(abs(v) for v in steps["cap"]["net_delta_pct"].values())
    band_tol = max(b["tol"] for b in tr["bands_pct"].values())
    assert cap_pct > band_tol, "cap did not move the book out of band; not the drop scenario"
    # The final book is back within the band and within BP.
    final = steps["repair"]
    assert final["bp_used"] <= tr["bp_limit"] + 1e-6
    for u, b in tr["bands_pct"].items():
        assert abs(final["net_delta_pct"][u] - b["target"]) <= b["tol"] + 1e-6


def test_cardinality_filter_never_drops_equity() -> None:
    """Equity is excluded from the per-underlying cap. With every candidate nonzero and
    max_positions=1, the cap zeros only options and leaves both equity legs."""
    req = _request()
    dvs = req["decision_variables"]
    groups = optimizer._group_by_underlying(dvs)
    bundle = optimizer._extract_inputs(
        dvs, req["per_underlying"], req["policy"], req["capital"], groups, req["horizon_days"])
    q = [1] * len(dvs)
    ranking = optimizer._compute_contributions(bundle, q)
    _q, dropped = optimizer._cardinality_filter(groups, q, bundle, ranking)
    is_equity = bundle["is_equity"]
    assert not any(is_equity[i] for i in dropped), "the cap dropped an equity leg"
    assert all(_q[i] == 1 for i in np.flatnonzero(is_equity)), "an equity leg was zeroed"
    # Options trimmed to the cap: one ZED option kept, the other dropped.
    kept_opts = [i for i in range(len(dvs)) if _q[i] and not is_equity[i]]
    assert len(kept_opts) == 1
