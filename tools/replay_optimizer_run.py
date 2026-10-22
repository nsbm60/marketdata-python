#!/usr/bin/env python3
"""Replay a captured optimizer request through the solve and the post-solve pipeline and
print the buying power consumed and the per-underlying net delta after each step:

    solve → netting → rounding → cap → repair

A "<== +N BP" marker flags the step that adds margin; "OVER" flags a book above the BP
limit. This is how you see whether the cap or the stock repair breaks the BP the
continuous solve respected.

    python tools/replay_optimizer_run.py tools/captured_requests/<request_id>.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import optimizer  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python tools/replay_optimizer_run.py <request.json>")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    tr = optimizer.pipeline_trace(payload)

    print(f"status={tr['status']}   bp_limit={tr['bp_limit']:.2f}   net_liq={tr['net_liq']:.2f}")
    print("bands (% of net liq):")
    for u, b in tr["bands_pct"].items():
        print(f"  {u:<8} target {b['target']:+.2f}  tol ±{b['tol']:.2f}")
    print()

    unds = list(tr["bands_pct"].keys())
    print(f"{'step':<9}{'bp_used':>12}   " + "   ".join(f"{u:>16}" for u in unds))
    print(f"{'':<9}{'':>12}   " + "   ".join(f"{'Δshares / %netliq':>16}" for _ in unds))
    prev_bp = None
    for s in tr["steps"]:
        bp = s["bp_used"]
        cells = []
        for u in unds:
            cells.append(f"{s['net_delta_shares'][u]:+8.1f} /{s['net_delta_pct'][u]:+5.1f}%")
        mark = ""
        if prev_bp is not None and bp - prev_bp > 1e-6:
            mark += f"   <== +{bp - prev_bp:.2f} BP"
        if bp > tr["bp_limit"] + 1e-6:
            mark += "   OVER"
        print(f"{s['step']:<9}{bp:>12.2f}   " + "   ".join(f"{c:>16}" for c in cells) + mark)
        prev_bp = bp


if __name__ == "__main__":
    main()
