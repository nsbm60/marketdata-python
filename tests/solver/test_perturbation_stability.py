"""The post-solve pipeline must be stable under a small input perturbation.

Run 87e615b0 came back infeasible-over-BP while 068fe199, the same book an hour later with
net liq ~2.8% lower, came back feasible. The flip was not a real portfolio change — it was
the cardinality cap (a discontinuous step: a dropped hedge swings net delta) handing the
band to an unbounded stock repair. A tiny input move tipped the book between feasible and
over-BP.

`solver_request_perturb.json` reproduces the mechanism deterministically: a balanced new
strangle on PERT with max_positions = 1, so the cap keeps one leg and drops the other,
swinging net delta to +44% before the re-solve re-hedges it back inside the ±8% band at the
BP limit. This test perturbs the inputs the solver consumes and asserts the result does not
flip and BP does not jump.

Perturbation levers. Spot ±0.5% is exact — it moves the band (shares) and the net-delta %.
IV is a diagnostic passthrough at the solver boundary (the solver consumes pre-computed
per-unit greeks and margins, not IV), so a ±0.2 vol move is modelled here as a small
proportional jitter of each option's per-unit delta/gamma/theta/margin — what an IV move
shifts downstream. Equity delta stays exactly ±1 (structural, per the spec).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import optimizer  # noqa: E402

_REQ = Path(__file__).parent / "solver_request_perturb.json"
_SPOT = (0.995, 1.0, 1.005)        # ±0.5%
_GREEK = (0.98, 1.0, 1.02)         # IV ±0.2 vol proxy on per-unit greeks/margin


def _base() -> dict:
    return json.loads(_REQ.read_text(encoding="utf-8"))


def _perturb(req: dict, spot_mult: float, greek_mult: float) -> dict:
    r = copy.deepcopy(req)
    for pu in r["per_underlying"]:
        pu["spot"] *= spot_mult
    for dv in r["decision_variables"]:
        if dv["type"] == "option":
            for f in ("delta_per_unit", "gamma_per_unit", "theta_per_unit", "margin_per_unit"):
                dv[f] *= greek_mult
    return r


def _grid():
    base = _base()
    for sm in _SPOT:
        for gm in _GREEK:
            yield sm, gm, optimizer.pipeline_trace(_perturb(base, sm, gm))


def _rounding_bp_bound(req: dict) -> float:
    """The most BP the stock repair can legitimately add: hedging a one-unit-per-position
    rounding swing of net delta with stock — Σ|option delta_per_unit| shares × equity margin."""
    opt_delta = sum(abs(dv["delta_per_unit"]) for dv in req["decision_variables"] if dv["type"] == "option")
    eq_margin = max(dv["margin_per_unit"] for dv in req["decision_variables"] if dv["type"] == "equity")
    return opt_delta * eq_margin


def test_cap_bites_in_the_base_case() -> None:
    """Guard the fixture: the base case must actually exercise the cardinality drop and the
    re-solve, or the stability test proves nothing."""
    tr = next(t for sm, gm, t in _grid() if sm == 1.0 and gm == 1.0)
    assert any(s["step"] == "resolve" for s in tr["steps"]), \
        "fixture no longer triggers the cap/re-solve path"


def test_result_does_not_flip_between_feasible_and_infeasible() -> None:
    statuses = {t["status"] for _sm, _gm, t in _grid()}
    assert len(statuses) == 1, f"status flipped across the perturbation grid: {statuses}"


def test_bp_never_exceeds_the_limit_and_the_repair_only_fixes_rounding() -> None:
    bound = _rounding_bp_bound(_base())
    for sm, gm, tr in _grid():
        steps = {s["step"]: s["bp_used"] for s in tr["steps"]}
        assert steps["repair"] <= tr["bp_limit"] + 1e-6, \
            f"spot×{sm} greek×{gm}: final BP {steps['repair']:.2f} > limit {tr['bp_limit']:.2f}"
        pre = steps.get("resolve_round", steps["cap"])
        assert steps["repair"] - pre <= bound + 1e-6, \
            f"spot×{sm} greek×{gm}: repair added {steps['repair'] - pre:.2f} BP > rounding bound {bound:.2f}"
