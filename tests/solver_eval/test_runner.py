"""Runner mechanics for the offline solver evaluation (spec step 1).

The per-case regression assertions live with the cases (step 5 scans every case with an
`expect` block); this file tests the harness itself: a promoted snapshot is a complete
request with the header, a `"snapshot"` case evaluates, portfolio application replaces the
book, a missing portfolio fails loudly, and perturbation is deterministic.
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


def test_portfolio_application_replaces_the_book() -> None:
    """empty flattens the snapshot's book; clean resolves to a small book on NVDA and MU only."""
    empty = se._resolve_case(_case("empty"))
    assert all(dv["current_quantity"] == 0 for dv in empty["decision_variables"])

    clean = se._resolve_case(_case("clean"))
    held = [dv for dv in clean["decision_variables"] if dv["current_quantity"]]
    assert held, "clean resolved to an empty book"
    assert {dv["underlying"] for dv in held} == {"NVDA", "MU"}
    # Two short option legs per name; each hedge is however many shares neutralise that name's
    # option delta (a balanced strangle needs ~0, so held equity legs are 0-2, not fixed).
    assert sum(1 for dv in held if dv["type"] == "option") == 4
    assert all(dv["direction"] == "short" for dv in held if dv["type"] == "option")
    assert sum(1 for dv in held if dv["type"] == "equity") <= 2


def test_missing_portfolio_file_fails_loudly() -> None:
    case = _case("live_20261022_5names")
    case["portfolio"] = "deep_itm_put"  # a spec-§3 portfolio not built yet
    with pytest.raises(SystemExit):
        se._resolve_case(case)


def test_perturbation_is_deterministic_and_moves_greeks() -> None:
    """A seeded perturbation reproduces exactly and actually nudges option greeks (anchored on
    the stored values, so a zero shock is a no-op and the leg's sign is preserved)."""
    import random

    req = se._resolve_case(_case("clean"))

    def _first_option_delta(seed: int) -> float:
        p = se._perturb_request(req, 0.5, 0.2, random.Random(seed))
        return next(dv["delta_per_unit"] for dv in p["decision_variables"]
                    if dv["type"] == "option" and dv["current_quantity"])

    base = next(dv["delta_per_unit"] for dv in req["decision_variables"]
                if dv["type"] == "option" and dv["current_quantity"])
    assert _first_option_delta(7) == _first_option_delta(7), "same seed must reproduce greeks"
    assert _first_option_delta(7) != base, "perturbation did not move the greek"
