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


if __name__ == "__main__":
    if sys.argv[1:] == ["--regenerate"]:
        _RESPONSE.write_text(json.dumps(_solve(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {_RESPONSE}")
    else:
        sys.exit("usage: python tests/solver/test_solver_golden.py --regenerate")
