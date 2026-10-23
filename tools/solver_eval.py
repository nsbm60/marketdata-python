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
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import optimizer  # noqa: E402
from greeks.domain import OptionRight  # noqa: E402
from greeks.solver import greeks as _bs_greeks  # noqa: E402

EVAL = ROOT / "tests" / "solver_eval"
MARKETS = EVAL / "markets"
PORTFOLIOS = EVAL / "portfolios"
CASES = EVAL / "cases"
SUITES = EVAL / "suites"
RESULTS = EVAL / "results"
CAPTURES = ROOT / "tools" / "captured_requests"

_EPS = 1e-6

# A contract's multiplier: per-unit greeks in the request are share-equivalent (raw greek ×
# this), matching optimizer.DELTA_RAW_DIVISOR. One place, so the perturbation recompute and
# the solver agree on the unit.
_MULTIPLIER = 100.0


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
    """The solver request for a case: the snapshot, with the test portfolio applied (if any)
    and policy overrides merged. Perturbation is a separate multi-run path (perturbation_runs);
    a single resolve is the unperturbed baseline."""
    request = _market_request(case["market"])
    portfolio_name = case.get("portfolio", "snapshot")
    if portfolio_name != "snapshot":
        portfolio = _load(PORTFOLIOS / f"{portfolio_name}.json")
        _apply_portfolio(request, portfolio, case["name"])
    _apply_policy(request, case.get("policy") or {})
    return request


# ── test portfolio resolution against a snapshot (spec §3) ───────────────────

def _right_from_id(dv_id: str) -> str:
    """The option right ('C'/'P') from a wire id: OSI stem + direction suffix, e.g.
    'NVDA261102C00160000-long'. The right is the letter before the 8-digit strike."""
    stem = dv_id.rsplit("-", 1)[0]
    return stem[-9]


def _apply_portfolio(request: dict, portfolio: dict, case_name: str) -> list:
    """Apply a test portfolio to the snapshot request: resolve each template line against the
    snapshot's own candidates and set current_quantity from the resolved book, every other
    record to zero (spec §3 — applying replaces the book). Resolution is deterministic because
    the snapshot is fixed. A line that can't resolve fails the case, naming the line. Returns
    one `resolved` note per line.

    net_liq: a number overrides the snapshot's; "snapshot" keeps it (the band is a % of it)."""
    dvs = request["decision_variables"]
    pu = {p["underlying"]: p for p in request["per_underlying"]}
    nl = portfolio.get("net_liq", "snapshot")
    if nl != "snapshot":
        request["capital"]["net_liq"] = float(nl)
    net_liq = float(request["capital"]["net_liq"])

    # Applying replaces the book: start from flat, then set the resolved positions.
    for dv in dvs:
        dv["current_quantity"] = 0

    notes = []
    option_lines = [p for p in portfolio.get("positions", []) if "shares" not in p]
    stock_lines = [p for p in portfolio.get("positions", []) if "shares" in p]

    for pos in option_lines:
        u, right, qty = pos["underlying"], pos["right"], int(pos["qty"])
        if qty == 0:
            raise ValueError(f"case {case_name}: {u} {right} line has qty 0")
        direction = "long" if qty > 0 else "short"
        cands = [dv for dv in dvs
                 if dv["type"] == "option" and dv["underlying"] == u
                 and dv["direction"] == direction and _right_from_id(dv["id"]) == right]
        if not cands:
            raise ValueError(f"case {case_name}: no {direction} {right} candidate for {u}")
        dte_target = float(pos["expiry"]["dte"])
        # Nearest listed expiry, the later one (larger dte) on a tie.
        best_dte = min({dv["days_to_expiry"] for dv in cands},
                       key=lambda d: (abs(d - dte_target), -d))
        at_expiry = [dv for dv in cands if dv["days_to_expiry"] == best_dte]
        chosen, how = _nearest_strike(at_expiry, pos["strike"], pu[u]["spot"], u, right, case_name)
        chosen["current_quantity"] = abs(qty)
        notes.append(f"{u} {right} qty {qty}: dte {dte_target} -> {best_dte:.2f}d; {how}; "
                     f"{chosen['id']}")

    for pos in stock_lines:
        u, shares = pos["underlying"], pos["shares"]
        spot = pu[u]["spot"]
        if "count" in shares:
            n = int(shares["count"])
            how = f"count {n}"
        elif "hedge_to_pct" in shares:
            p = float(shares["hedge_to_pct"])
            opt_delta = sum(dv["current_quantity"] * dv["delta_per_unit"]
                            for dv in dvs if dv["underlying"] == u and dv["type"] == "option")
            target_delta = p / 100.0 * net_liq / spot
            n = int(round(target_delta - opt_delta))
            how = (f"hedge_to_pct {p}: option delta {opt_delta:+.1f} sh, "
                   f"target {target_delta:+.1f} sh -> {n:+d} sh")
        else:
            raise ValueError(f"case {case_name}: {u} shares line needs count or hedge_to_pct")
        direction = "long" if n >= 0 else "short"
        leg = next((dv for dv in dvs if dv["type"] == "equity"
                    and dv["underlying"] == u and dv["direction"] == direction), None)
        if leg is None:
            raise ValueError(f"case {case_name}: no {direction} equity leg for {u}")
        leg["current_quantity"] = abs(n)
        notes.append(f"{u} stock: {how}")

    return notes


def _nearest_strike(cands: list, strike_spec: dict, spot: float, u: str, right: str,
                    case_name: str) -> tuple:
    """The candidate nearest the strike target (moneyness = strike/spot, delta = |Δ| from
    delta_per_unit, or offset = $ from spot), with a note of what it resolved to."""
    if "moneyness" in strike_spec:
        m = float(strike_spec["moneyness"])
        chosen = min(cands, key=lambda dv: abs(dv["strike"] / spot - m))
        return chosen, f"moneyness {m} -> strike {chosen['strike']} (spot {spot:.2f})"
    if "delta" in strike_spec:
        d = float(strike_spec["delta"])
        chosen = min(cands, key=lambda dv: abs(abs(dv["delta_per_unit"] / _MULTIPLIER) - d))
        return chosen, (f"delta {d} -> strike {chosen['strike']} "
                        f"(|Δ| {abs(chosen['delta_per_unit'] / _MULTIPLIER):.3f})")
    if "offset" in strike_spec:
        off = float(strike_spec["offset"])
        chosen = min(cands, key=lambda dv: abs((dv["strike"] - spot) - off))
        return chosen, f"offset {off:+} -> strike {chosen['strike']} (spot {spot:.2f})"
    raise ValueError(f"case {case_name}: {u} {right} strike needs moneyness, delta or offset")


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


# ── perturbation (spec §4) ───────────────────────────────────────────────────

def _perturb_request(request: dict, spot_pct: float, iv_points: float, rng: random.Random) -> dict:
    """One perturbed copy of the request: each underlying's spot nudged uniformly within
    ±spot_pct %, each underlying's IV shifted uniformly within ±iv_points vol points (one vol
    point = 0.01 of sigma, so 0.2 vol points is ±0.002 in decimal vol), both from the seeded
    rng. Every option's delta/gamma/theta is recomputed with Black-76 — not scaled — and
    *anchored*: the greek moves by the model-predicted change between (spot, iv) and (spot',
    iv'), so a zero shock leaves the stored greek exactly as captured and only the modelled
    sensitivity moves. Forward = spot, r = 0 (the change is first-order insensitive to the rate,
    so the baseline vendor greeks are not contaminated by a pricing-model mismatch). Margin is
    held fixed: a sub-1% spot move barely moves Reg-T, and the spec recomputes greeks and
    prices, not margin. The per-underlying spot (which the band is a % of) is updated."""
    req = copy.deepcopy(request)
    pu = {p["underlying"]: p for p in req["per_underlying"]}
    spot0 = {u: pu[u]["spot"] for u in pu}
    spot_shock = {u: rng.uniform(-spot_pct, spot_pct) / 100.0 for u in pu}
    iv_shift = {u: rng.uniform(-iv_points, iv_points) * 0.01 for u in pu}  # vol points → sigma
    for u in pu:
        pu[u]["spot"] = spot0[u] * (1.0 + spot_shock[u])
    for dv in req["decision_variables"]:
        if dv["type"] != "option":
            continue
        u = dv["underlying"]
        s0, s1 = spot0[u], pu[u]["spot"]
        iv0 = dv["iv"]
        iv1 = max(1e-6, iv0 + iv_shift[u])
        t = dv["days_to_expiry"] / 365.0
        if t <= 0.0:
            continue  # expiring today: intrinsic only, greeks already ~0
        right = OptionRight.CALL if _right_from_id(dv["id"]) == "C" else OptionRight.PUT
        g0 = _bs_greeks(s0, dv["strike"], t, 1.0, iv0, right, spot=s0)
        g1 = _bs_greeks(s1, dv["strike"], t, 1.0, iv1, right, spot=s1)
        sign = 1.0 if dv["direction"] == "long" else -1.0  # stored greek carries its leg's sign
        dv["delta_per_unit"] += sign * (g1.delta - g0.delta) * _MULTIPLIER
        dv["gamma_per_unit"] += sign * (g1.gamma - g0.gamma) * _MULTIPLIER
        dv["theta_per_unit"] += sign * (g1.theta - g0.theta) * _MULTIPLIER
        dv["iv"] = iv1
    return req


def _solve_summary(request: dict) -> dict:
    """One solve reduced to what perturbation compares: the recommended (optimal) book, its
    objective, the action, BP used and each name's net delta %."""
    resp = optimizer.handle_optimize(copy.deepcopy(request))
    trace = optimizer.pipeline_trace(copy.deepcopy(request))
    final = trace["steps"][-1]
    book = {r["id"]: r["optimal_quantity"] for r in resp.get("recommendations", [])
            if r.get("optimal_quantity", 0)}
    score = resp.get("score") or {}
    return {
        "status": resp.get("status"),
        "action": resp.get("action"),
        "objective": score.get("optimal_objective_value"),
        "book": book,
        "bp_used": final["bp_used"],
        "bp_limit": trace["bp_limit"],
        "net_delta_pct": final["net_delta_pct"],
        "resp": resp,
    }


def perturbation_runs(request: dict, perturb: dict) -> dict:
    """The baseline solve plus one solve per perturbation variant (spec §4). Deterministic for
    a given seed. Returns baseline and variant summaries for the caller to report the spread."""
    rng = random.Random(perturb["seed"])
    spot_pct = float(perturb["spot_pct"])
    iv_pts = float(perturb["iv_vol_points"])
    n = int(perturb["variants"])
    baseline = _solve_summary(request)
    variants = []
    for _ in range(n):
        preq = _perturb_request(request, spot_pct, iv_pts, rng)
        variants.append(_solve_summary(preq))
    return {"baseline": baseline, "variants": variants,
            "spot_pct": spot_pct, "iv_vol_points": iv_pts, "seed": perturb["seed"], "n": n}


def _stability(request: dict, perturb: dict) -> dict:
    """expect.stable support: no variant flips between feasible and infeasible, and BP varies by
    no more than one contract's margin (the rounding grain)."""
    runs = perturbation_runs(request, perturb)
    summaries = [runs["baseline"], *runs["variants"]]
    infeasible = {s["status"] == "infeasible" for s in summaries}
    bps = [s["bp_used"] for s in summaries]
    return {"no_feasibility_flip": len(infeasible) == 1, "bp_spread": max(bps) - min(bps),
            "runs": runs}


# ── case evaluation ──────────────────────────────────────────────────────────

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
        s = m["stability"]
        out.append(("stable", s["no_feasibility_flip"] == expect["stable"],
                    f"feasibility_flip={not s['no_feasibility_flip']} bp_spread={s['bp_spread']:.2f}"))
    return out


def evaluate_case(case: dict) -> dict:
    """Resolve and run one case. Returns metrics plus the expectation checks. Importable by
    the regression tests."""
    request = _resolve_case(case)
    m = _metrics(request)
    expect = case.get("expect") or {}
    if "stable" in expect:
        m["stability"] = _stability(request, case["perturb"])
    m["checks"] = _check_expect(expect, m)
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
