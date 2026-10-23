#!/usr/bin/env python3
"""Readable baseline report for the solver evaluation (spec §3–4), for the `empty` and `clean`
cases against the live_20261022_5names snapshot. Not a pass/fail suite — it prints the full
picture for a human to read:

  - repeatability: each case solved twice, the two solves must be identical;
  - empty: the whole recommended book (contract, qty, theta, exec cost, gamma), BP used, and
    each name's delta against its band;
  - clean: the action, the optimal-vs-current gap, and every trade with its contributions;
  - perturbation: spot ±0.5% / IV ±0.2 vol points, 20 variants, and how far the book and the
    objective move.

    python tools/eval_baseline_report.py
"""
from __future__ import annotations

import copy
import logging
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optimizer  # noqa: E402
import tools.solver_eval as se  # noqa: E402

# Offline report: don't capture requests to disk, and keep the solver's INFO chatter out of
# the output (warnings and above still show).
optimizer._capture_request = lambda *a, **k: None  # noqa: SLF001
logging.getLogger().setLevel(logging.WARNING)

RULE = "=" * 78
SUB = "-" * 78


def _load_case(name: str) -> dict:
    import json
    return json.loads((se.CASES / f"{name}.json").read_text(encoding="utf-8"))


def _band_line(trace_or_summary: dict, request: dict) -> list:
    """Each name's net delta % against its band, as readable lines."""
    pu = {p["underlying"]: p for p in request["per_underlying"]}
    out = []
    for u, pct in trace_or_summary["net_delta_pct"].items():
        tgt = pu[u]["delta_target_pct"]
        tol = pu[u]["delta_tolerance_pct"]
        lo, hi = tgt - tol, tgt + tol
        mark = "in-band" if lo - 1e-6 <= pct <= hi + 1e-6 else "OUT OF BAND"
        out.append(f"      {u:5}  {pct:+7.2f}%   band [{lo:+.0f}%, {hi:+.0f}%]   {mark}")
    return out


def _solve(request: dict) -> dict:
    return se._solve_summary(request)  # status, action, objective, book, bp_used, net_delta_pct, resp


def _recs_by_id(resp: dict) -> dict:
    return {r["id"]: r for r in resp.get("recommendations", [])}


def repeatability(name: str, request: dict) -> None:
    a = _solve(copy.deepcopy(request))
    b = _solve(copy.deepcopy(request))
    same_obj = a["objective"] == b["objective"]
    same_book = a["book"] == b["book"]
    print(f"  {name}: objective {a['objective']!r} vs {b['objective']!r} -> "
          f"{'IDENTICAL' if same_obj else 'DIFFERENT'}")
    print(f"  {name}: book ({len(a['book'])} positions) -> "
          f"{'IDENTICAL' if same_book else 'DIFFERENT'}")
    if not (same_obj and same_book):
        print(f"  {name}: *** NOT REPEATABLE ***")


def report_empty(request: dict) -> None:
    s = _solve(request)
    recs = _recs_by_id(s["resp"])
    print(f"  status={s['status']}  action={s['action']}  objective={s['objective']:.2f}  "
          f"BP used={s['bp_used']:.0f} / {s['bp_limit']:.0f} limit "
          f"({s['bp_used'] / s['bp_limit'] * 100:.0f}%)")
    book = {i: q for i, q in s["book"].items()}
    print(f"  recommended book: {len(book)} positions")
    print(f"      {'contract':32} {'qty':>5} {'theta':>10} {'exec_cost':>11} {'gamma':>10} {'total':>10}")
    for cid in sorted(book, key=lambda c: -recs[c]["objective_contribution"]["total"]):
        r = recs[cid]
        oc = r["objective_contribution"]
        print(f"      {cid:32} {r['optimal_quantity']:>5} {oc['theta']:>10.2f} "
              f"{oc['exec_cost']:>11.2f} {oc['gamma']:>10.2f} {oc['total']:>10.2f}")
    print("  delta vs band:")
    for line in _band_line(s, request):
        print(line)


def report_clean(request: dict) -> None:
    s = _solve(request)
    score = s["resp"]["score"]
    recs = s["resp"]["recommendations"]
    gap = score["optimal_objective_value"] - score["current_objective_value"]
    print(f"  status={s['status']}  action={s['action']}")
    print(f"  current objective = {score['current_objective_value']:.2f}")
    print(f"  optimal objective = {score['optimal_objective_value']:.2f}")
    print(f"  gap (optimal - current) = {gap:.2f}   (min_improvement = "
          f"{request['policy']['min_improvement']})")
    trades = [r for r in recs if r["target_quantity"] != r["current_quantity"]]
    print(f"  trades (target != current): {len(trades)}")
    if trades:
        print(f"      {'contract':32} {'cur':>4} {'->':>3} {'tgt':>4} {'theta':>9} "
              f"{'exec':>8} {'gamma':>9} {'total':>9}")
        for r in trades:
            oc = r["objective_contribution"]
            print(f"      {r['id']:32} {r['current_quantity']:>4} {'->':>3} "
                  f"{r['target_quantity']:>4} {oc['theta']:>9.2f} {oc['exec_cost']:>8.2f} "
                  f"{oc['gamma']:>9.2f} {oc['total']:>9.2f}")
    print("  delta vs band:")
    for line in _band_line(s, request):
        print(line)


def report_perturbation(name: str, request: dict, perturb: dict) -> None:
    runs = se.perturbation_runs(request, perturb)
    base = runs["baseline"]
    variants = runs["variants"]
    print(f"  {name}: spot ±{runs['spot_pct']}% , IV ±{runs['iv_vol_points']} vol points, "
          f"{runs['n']} variants, seed {runs['seed']}")

    feasible = [v for v in variants if v["objective"] is not None]
    infeasible = [v for v in variants if v["objective"] is None]
    b = base["objective"]

    statuses = {v["status"] for v in variants} | {base["status"]}
    print(f"    status across all runs: {sorted(statuses)}  "
          f"(baseline={base['status']}, {len(infeasible)}/{len(variants)} variants infeasible)")

    print(f"    objective: baseline {b:.2f}" if b is not None
          else "    objective: baseline INFEASIBLE")
    if feasible:
        objs = [v["objective"] for v in feasible]
        print(f"      feasible variants ({len(feasible)}): min {min(objs):.2f}  max {max(objs):.2f}  "
              f"mean {statistics.mean(objs):.2f}  stdev {statistics.pstdev(objs):.2f}")
        if b:
            swing = max(abs(min(objs) - b), abs(max(objs) - b))
            print(f"      max swing from baseline: {swing:.2f}  ({swing / abs(b) * 100:.1f}% of baseline)")

    # Book movement vs the baseline book, feasible variants only (an infeasible variant has no
    # recommended book to compare).
    base_book = base["book"]
    moved_counts, moved_qty = [], []
    for v in feasible:
        ids = set(base_book) | set(v["book"])
        moved_counts.append(sum(1 for i in ids if base_book.get(i, 0) != v["book"].get(i, 0)))
        moved_qty.append(sum(abs(base_book.get(i, 0) - v["book"].get(i, 0)) for i in ids))
    print(f"    book movement vs baseline ({len(base_book)} positions at baseline, "
          f"feasible variants only):")
    if moved_counts:
        identical = sum(1 for c in moved_counts if c == 0)
        print(f"      contracts changed per variant: min {min(moved_counts)}  "
              f"max {max(moved_counts)}  mean {statistics.mean(moved_counts):.1f}")
        print(f"      total |Δqty| per variant:      min {min(moved_qty)}  "
              f"max {max(moved_qty)}  mean {statistics.mean(moved_qty):.1f}")
        print(f"      variants with an identical book to baseline: {identical}/{len(feasible)}")


def main() -> None:
    empty_case = _load_case("empty")
    clean_case = _load_case("clean")
    empty_req = se._resolve_case(empty_case)
    clean_req = se._resolve_case(clean_case)

    print(RULE)
    print("SOLVER EVALUATION — baseline report  (market: live_20261022_5names = 068fe199)")
    print(f"solver commit {se._solver_commit()}  cvxpy {optimizer.cp.__version__}")
    print(RULE)

    print("\n[1] REPEATABILITY — each case solved twice, must be identical")
    print(SUB)
    repeatability("empty", empty_req)
    repeatability("clean", clean_req)

    print("\n[2] EMPTY — built from a clean sheet")
    print(SUB)
    report_empty(empty_req)

    print("\n[3] CLEAN — a hedged book inside every limit; does it hold?")
    print(SUB)
    report_clean(clean_req)

    print("\n[4] PERTURBATION — how far the book and objective move")
    print(SUB)
    report_perturbation("empty", empty_req, empty_case["perturb"])
    print()
    report_perturbation("clean", clean_req, clean_case["perturb"])
    print("\n" + RULE)


if __name__ == "__main__":
    main()
