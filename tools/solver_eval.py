#!/usr/bin/env python3
"""Offline solver evaluation — docs/specs/solver-evaluation-spec-draft.md.

A test is a frozen market (a saved solver request) plus a test portfolio plus a policy, run
through the solver with no CalcServer, no MDS and no live data. Every test reruns exactly on
any solver version, so a failure can be reproduced and a fix proved — unlike a live run,
whose inputs drift between calls.

Commands (build order §8; this file covers steps 1–2):
  snapshot <request_id> <name> [notes]   promote a captured live request to a committed market
  run <case|suite> [--trace]             run case(s), print a line each, check expectations

`portfolio` resolution against a snapshot (step 3) is not built yet: a case must use
`"portfolio": "snapshot"`. Perturbation / `expect.stable` (step 4) is not built yet either.
Both raise rather than silently skip.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import optimizer  # noqa: E402

EVAL = ROOT / "tests" / "solver_eval"
MARKETS = EVAL / "markets"
CASES = EVAL / "cases"
SUITES = EVAL / "suites"
RESULTS = EVAL / "results"
CAPTURES = ROOT / "tools" / "captured_requests"

_EPS = 1e-6


def _load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _solver_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ── snapshot promotion (spec §2) ─────────────────────────────────────────────

def cmd_snapshot(request_id: str, name: str, notes: str) -> None:
    src = CAPTURES / f"{request_id}.json"
    req = _load(src)
    dest = MARKETS / f"{name}.json"
    if dest.exists():
        sys.exit(f"snapshot already exists (never edited after commit): {dest}")
    # The header is additive; the request below it is untouched (a snapshot is a complete
    # solver request). captured_at is the market's own timestamp, not wall-clock now.
    out = {
        "snapshot_name": name,
        "source": "captured",
        "captured_at": req.get("timestamp"),
        "notes": notes,
        **req,
    }
    MARKETS.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"promoted {request_id} -> {dest}  ({len(req['decision_variables'])} vars, "
          f"net_liq={req['capital']['net_liq']})")


# ── case resolution and evaluation (spec §4–5) ───────────────────────────────

def _market_request(market_name: str) -> dict:
    """The committed snapshot as a plain solver request (header stripped)."""
    snap = _load(MARKETS / f"{market_name}.json")
    return {k: v for k, v in snap.items()
            if k not in ("snapshot_name", "source", "captured_at", "notes")}


def _apply_policy(request: dict, overrides: dict) -> None:
    """Merge case policy overrides onto the snapshot's policy (spec §4)."""
    if overrides:
        request["policy"] = {**request["policy"], **overrides}


def _resolve_case(case: dict) -> dict:
    request = _market_request(case["market"])
    portfolio = case.get("portfolio", "snapshot")
    if portfolio != "snapshot":
        raise NotImplementedError(
            f"case {case['name']}: portfolio resolution is spec step 3, not built; "
            f"use \"portfolio\": \"snapshot\"")
    if case.get("perturb") is not None:
        raise NotImplementedError(
            f"case {case['name']}: perturbation is spec step 4, not built")
    _apply_policy(request, case.get("policy") or {})
    return request


def _metrics(request: dict) -> dict:
    """Run the solver and derive the summary the spec's run line reports. Uses
    handle_optimize for the response and pipeline_trace for BP and per-name net delta."""
    resp = optimizer.handle_optimize(copy.deepcopy(request))
    trace = optimizer.pipeline_trace(copy.deepcopy(request))
    final = trace["steps"][-1]
    bp_used, bp_limit = final["bp_used"], trace["bp_limit"]
    band_breaches = []
    for u, b in trace["bands_pct"].items():
        pct = final["net_delta_pct"][u]
        if abs(pct - b["target"]) > b["tol"] + _EPS:
            band_breaches.append(f"{u} {pct:+.2f}% (±{b['tol']:.2f})")
    recs = resp.get("recommendations", [])
    trades = sum(1 for r in recs if r["target_quantity"] != r["current_quantity"])
    return {
        "status": resp.get("status"),
        "action": resp.get("action"),
        "bp_used": bp_used, "bp_limit": bp_limit,
        "bp_within": bp_used <= bp_limit + _EPS,
        "band_breaches": band_breaches,
        "bands_within": not band_breaches,
        "trades": trades,
        "resp": resp, "trace": trace,
    }


def _check_expect(expect: dict, m: dict) -> list:
    """(name, ok, detail) per assertion in the case's expect block (spec §4)."""
    out = []
    if "status" in expect:
        out.append(("status", m["status"] == expect["status"], m["status"]))
    if "status_not" in expect:
        out.append(("status_not", m["status"] not in expect["status_not"], m["status"]))
    if "action" in expect:
        out.append(("action", m["action"] == expect["action"], m["action"]))
    if "bp_within_limit" in expect:
        out.append(("bp_within_limit", m["bp_within"] == expect["bp_within_limit"],
                    f"{m['bp_used']:.2f}/{m['bp_limit']:.2f}"))
    if "bands_within" in expect:
        out.append(("bands_within", m["bands_within"] == expect["bands_within"],
                    ",".join(m["band_breaches"]) or "ok"))
    if "max_trades" in expect:
        out.append(("max_trades", m["trades"] <= expect["max_trades"], m["trades"]))
    if "stable" in expect:
        raise NotImplementedError("expect.stable needs perturbation (spec step 4), not built")
    return out


def evaluate_case(case: dict) -> dict:
    """Resolve and run one case. Returns metrics plus the expectation checks. Importable by
    the regression tests."""
    request = _resolve_case(case)
    m = _metrics(request)
    m["checks"] = _check_expect(case.get("expect") or {}, m)
    m["request"] = request
    return m


def _print_case(case: dict, m: dict, trace: bool) -> bool:
    checks = m["checks"]
    passed = all(ok for _n, ok, _d in checks)
    band = ",".join(m["band_breaches"]) or "in-band"
    verdict = "" if not checks else ("  PASS" if passed else "  FAIL")
    print(f"[{case['name']}] {case.get('tests','')}")
    print(f"    status={m['status']} action={m['action']} "
          f"bp={m['bp_used']:.0f}/{m['bp_limit']:.0f} {band} trades={m['trades']}{verdict}")
    for n, ok, d in checks:
        if not ok:
            print(f"      ✗ {n}: {d}")
    if trace:
        for s in m["trace"]["steps"]:
            print(f"      {s['step']:<13} bp={s['bp_used']:.0f}")
    return passed


def _cases_for(target: str) -> list:
    if (CASES / f"{target}.json").exists():
        return [target]
    if (SUITES / f"{target}.json").exists():
        return _load(SUITES / f"{target}.json")
    sys.exit(f"no case or suite named {target!r}")


def cmd_run(target: str, trace: bool) -> None:
    names = _cases_for(target)
    all_ok = True
    for name in names:
        case = _load(CASES / f"{name}.json")
        m = evaluate_case(case)
        all_ok &= _print_case(case, m, trace)
    print(f"\nsolver {_solver_commit()}  cvxpy {optimizer.cp.__version__}")
    sys.exit(0 if all_ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline solver evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="promote a captured request to a committed market")
    s.add_argument("request_id")
    s.add_argument("name")
    s.add_argument("notes", nargs="?", default="")
    r = sub.add_parser("run", help="run a case or suite")
    r.add_argument("target")
    r.add_argument("--trace", action="store_true")
    a = ap.parse_args()
    if a.cmd == "snapshot":
        cmd_snapshot(a.request_id, a.name, a.notes)
    elif a.cmd == "run":
        cmd_run(a.target, a.trace)


if __name__ == "__main__":
    main()
