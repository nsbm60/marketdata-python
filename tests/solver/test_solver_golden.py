"""Cross-language golden contract for the optimizer request and response.

`solver_request_golden.json` is what the Scala `SolverRequestEncoder` produces for a fixed
problem spec (MarketData `SolverRequestGoldenSpec`). It is committed byte-identically here
and in MarketData `src/test/resources/`, together with `solver_response_golden.json`.

This test runs the solver on the request fixture and checks the result against the response
fixture. The Scala spec checks the encoder against the request fixture and its decoder
against the response fixture. The two repos' copies ARE the contract.

To change the format or the formulation on purpose, regenerate the request from Scala
(`sbt "Test/runMain com.nsbm.marketdata.calc.optimizer.SolverRequestGolden"`), then the
response here:

    python tests/solver/test_solver_golden.py --regenerate

and copy both files to both repos.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

# Repo root on the path, so `--regenerate` also works when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import optimizer  # noqa: E402

_HERE = Path(__file__).parent
_REQUEST = _HERE / "solver_request_golden.json"
_RESPONSE = _HERE / "solver_response_golden.json"

# Run-dependent: differ on every solve, so not part of the contract.
_VOLATILE = {("response_timestamp",), ("solver_meta", "wall_clock_seconds"), ("solver_meta", "iterations")}
_REL_TOL = 1e-6
_ABS_TOL = 1e-6


def _solve() -> dict:
    request = json.loads(_REQUEST.read_text(encoding="utf-8"))
    return json.loads(json.dumps(optimizer.handle_optimize(copy.deepcopy(request))))


def _diff(expected, actual, path=()) -> list[str]:
    if path in _VOLATILE:
        return []
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            if (path + (key,)) in _VOLATILE:
                continue
            if key not in expected or key not in actual:
                out.append(f"{'.'.join(path + (key,))}: present on one side only")
            else:
                out += _diff(expected[key], actual[key], path + (key,))
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{'.'.join(path)}: length {len(expected)} != {len(actual)}"]
        out = []
        for i, (e, a) in enumerate(zip(expected, actual)):
            out += _diff(e, a, path + (str(i),))
        return out
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) \
            and not isinstance(expected, bool) and not isinstance(actual, bool):
        if math.isclose(expected, actual, rel_tol=_REL_TOL, abs_tol=_ABS_TOL):
            return []
        return [f"{'.'.join(path)}: {expected!r} != {actual!r}"]
    return [] if expected == actual else [f"{'.'.join(path)}: {expected!r} != {actual!r}"]


def test_solver_reproduces_golden_response() -> None:
    expected = json.loads(_RESPONSE.read_text(encoding="utf-8"))
    differences = _diff(expected, _solve())
    assert not differences, "solver output differs from the golden response:\n" + "\n".join(differences)


def test_golden_request_decodes_and_solves_optimal() -> None:
    assert _solve()["status"] == "optimal"


def _golden() -> tuple[dict, dict]:
    request = json.loads(_REQUEST.read_text(encoding="utf-8"))
    response = json.loads(_RESPONSE.read_text(encoding="utf-8"))
    return request, response


def test_golden_book_is_inside_every_delta_band() -> None:
    """The golden continuous optimum sits on the band edges (NVDA +25 shares, MU −25: the
    2.15% and 1.18% bands of $250k at their spots); the rounded book must still be inside
    every band, checked in % of net liq. (The case where rounding crosses the
    edge and stock repairs it is tools/test_real_optimizer.py, "Rounding: Stock Repair".)"""
    request, response = _golden()
    dvs = {dv["id"]: dv for dv in request["decision_variables"]}
    net: dict[str, float] = {}
    for r in response["recommendations"]:
        net[r["underlying"]] = net.get(r["underlying"], 0.0) + dvs[r["id"]]["delta_per_unit"] * r["optimal_quantity"]
    net_liq = request["capital"]["net_liq"]
    for pu in request["per_underlying"]:
        u = pu["underlying"]
        pct = pu["spot"] * net.get(u, 0.0) / net_liq * 100.0
        assert abs(pct - pu["delta_target_pct"]) <= pu["delta_tolerance_pct"] + 1e-9, \
            f"{u} dollar delta {pct:.4f}% outside {pu['delta_target_pct']} ± {pu['delta_tolerance_pct']}%"


def test_golden_risk_penalty_is_the_dollar_variance_overlap_form() -> None:
    """The golden book has a same-day MU call next to a 5-day-horizon MU short: the dollar
    P&L-variance risk penalty must equal −(A/2)/net_liq · Σ_U 2δ·σ_U⁴·Σ_i Σ_j cᵢcⱼ·min(hᵢ,hⱼ),
    with cᵢ = ½·γ_i·S_U²·qᵢ the dollar gamma, σ_U the underlying's ATM IV, δ the hedge interval
    in years, and hᵢ in years — hedge credit only for the overlap (amendment 2026-10-23)."""
    request, response = _golden()
    dvs = {dv["id"]: dv for dv in request["decision_variables"]}
    horizon = request["horizon_days"]
    A = request["policy"]["risk_aversion"]
    net_liq = request["capital"]["net_liq"]
    delta_years = request.get("hedge_interval_days", 1.0) / 252.0
    spot = {pu["underlying"]: pu["spot"] for pu in request["per_underlying"]}
    atm_iv = {}
    for pu in request["per_underlying"]:
        opts = [dv for dv in request["decision_variables"]
                if dv["type"] == "option" and dv["underlying"] == pu["underlying"]]
        atm_iv[pu["underlying"]] = (
            min(opts, key=lambda dv: abs(dv["strike"] - pu["spot"]))["iv"] if opts else 0.0)
    legs = []
    for r in response["recommendations"]:
        dv = dvs[r["id"]]
        u = r["underlying"]
        h = (min(dv["days_to_expiry"], horizon) if dv["type"] == "option" else horizon) / 365.0
        c = 0.5 * dv["gamma_per_unit"] * spot[u] ** 2 * r["target_quantity"]
        legs.append((u, c, h))
    r_u: dict[str, float] = {}
    for (ui, ci, hi) in legs:
        for (uj, cj, hj) in legs:
            if ui == uj:
                r_u[ui] = r_u.get(ui, 0.0) + ci * cj * min(hi, hj)
    expected = -(A / 2.0) / net_liq * sum(2.0 * delta_years * atm_iv[u] ** 4 * r_u[u] for u in r_u)
    got = response["score"]["components"]["risk_penalty"]
    assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9), f"{got} != {expected}"


def test_golden_book_never_holds_both_legs_of_an_instrument() -> None:
    _request, response = _golden()
    held = {r["id"] for r in response["recommendations"] if r["optimal_quantity"] > 0}
    for rid in held:
        if rid.endswith("-long"):
            assert rid[: -len("-long")] + "-short" not in held, f"{rid} held long and short"


if __name__ == "__main__":
    if sys.argv[1:] == ["--regenerate"]:
        _RESPONSE.write_text(json.dumps(_solve(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {_RESPONSE}")
    else:
        sys.exit("usage: python tests/solver/test_solver_golden.py --regenerate")
