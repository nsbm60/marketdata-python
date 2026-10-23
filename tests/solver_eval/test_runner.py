"""Runner mechanics for the offline solver evaluation (spec step 1).

The per-case regression assertions live with the cases (step 5 scans every case with an
`expect` block); this file tests the harness itself: a promoted snapshot is a complete
request with the header, a `"snapshot"` case evaluates, and the not-yet-built resolution
paths fail loudly rather than silently passing.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parents[1]))
import tools.solver_eval as se  # noqa: E402


def _case(name: str) -> dict:
    return json.loads((_HERE / "cases" / f"{name}.json").read_text(encoding="utf-8"))


def test_promoted_snapshot_has_header_and_is_a_request() -> None:
    snap = json.loads((_HERE / "markets" / "live_20261022_5names.json").read_text(encoding="utf-8"))
    for k in ("snapshot_name", "source", "captured_at", "notes"):
        assert k in snap, f"snapshot missing header field {k}"
    assert snap["source"] == "captured"
    # Still a complete solver request below the header.
    for k in ("decision_variables", "per_underlying", "policy", "capital", "horizon_days"):
        assert k in snap, f"snapshot missing request field {k}"


def test_snapshot_case_evaluates_and_meets_expectations() -> None:
    m = se.evaluate_case(_case("live_20261022_5names"))
    assert m["status"] == "optimal"
    assert all(ok for _n, ok, _d in m["checks"]), [c for c in m["checks"] if not c[1]]


def test_unbuilt_portfolio_resolution_raises() -> None:
    case = _case("live_20261022_5names")
    case["portfolio"] = "deep_itm_put"
    with pytest.raises(NotImplementedError):
        se.evaluate_case(case)


def test_unbuilt_perturbation_raises() -> None:
    case = _case("live_20261022_5names")
    case["perturb"] = {"spot_pct": 0.5, "iv_vol_points": 0.2, "seed": 1, "variants": 5}
    with pytest.raises(NotImplementedError):
        se.evaluate_case(case)
