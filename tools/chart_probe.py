#!/usr/bin/env python3
"""
chart_probe.py

Dump what CalcServer actually returns for a chart request: the bars at the
edges, the largest true ranges, and the shape of the ATR series.

Answers three questions the chart itself cannot:
  - is the first bar of the window an outlier, and is it plausible?
  - does the ATR series cover the bars, or stop short?
  - is the ATR pane flat because the values are flat, or because one spike
    owns the scale?

Usage:
    python tools/chart_probe.py --symbol NVDA --timeframe 1m --bars 543
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from control_client import ControlTimeout, request as control_request
from discovery import ControlService, ServiceLocator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NVDA")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--bars", type=int, default=543)
    ap.add_argument("--session", default="extended")
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--timeout", type=int, default=20000)
    args = ap.parse_args()

    calc = ServiceLocator.control(ControlService.CALC, timeout_sec=30)

    request = {
        "op": "get_chart_data",
        "symbol": args.symbol.upper(),
        "timeframe": args.timeframe,
        "barCount": args.bars,
        "atrPeriod": args.atr_period,
    }
    if args.session:
        request["session"] = args.session

    try:
        reply = control_request(calc.router, request, timeout_ms=args.timeout)
    except ControlTimeout:
        print("No reply from CalcServer.")
        return 1

    if not reply.get("ok", False):
        print(f"Request failed: {reply.get('error')}")
        return 1

    data = reply.get("data", {})
    if "data" in data and isinstance(data["data"], dict):
        data = data["data"]

    bars = data.get("bars") or []
    atr = data.get("atr") or []
    if not bars:
        print("No bars returned.")
        return 1

    print(f"{args.symbol.upper()} {args.timeframe} session={args.session}")
    print(f"bars: {len(bars)}   atr points: {len(atr)}   "
          f"(expected {max(0, len(bars) - args.atr_period)})")
    print()

    print("First 5 bars:")
    for b in bars[:5]:
        print(f"  {b['t']}  o={b['o']:<9.4f} h={b['h']:<9.4f} l={b['l']:<9.4f} "
              f"c={b['c']:<9.4f} v={b.get('v', 0):<10} range={b['h'] - b['l']:.4f}")
    print("Last 2 bars:")
    for b in bars[-2:]:
        print(f"  {b['t']}  o={b['o']:<9.4f} h={b['h']:<9.4f} l={b['l']:<9.4f} "
              f"c={b['c']:<9.4f} v={b.get('v', 0):<10} range={b['h'] - b['l']:.4f}")
    print()

    trs = []
    for i in range(1, len(bars)):
        prev, cur = bars[i - 1], bars[i]
        tr = max(cur["h"] - cur["l"],
                 abs(cur["h"] - prev["c"]),
                 abs(cur["l"] - prev["c"]))
        trs.append((tr, i, cur["t"]))

    ordered = sorted(trs, reverse=True)
    median = sorted(t[0] for t in trs)[len(trs) // 2]
    print(f"True range: median {median:.4f}")
    print("  largest 5:")
    for tr, i, t in ordered[:5]:
        print(f"    bar {i:<5} {t}  TR={tr:.4f}   ({tr / median:.0f}x median)")
    print(f"  how many of the first {args.atr_period} true ranges are >10x median: "
          f"{sum(1 for tr, i, _ in trs[:args.atr_period] if tr > 10 * median)}")
    print()

    if atr:
        vals = [p["atr"] for p in atr]
        print(f"ATR series: first {vals[0]:.4f}   last {vals[-1]:.4f}   "
              f"min {min(vals):.4f}   max {max(vals):.4f}")
        print(f"  first value is {vals[0] / vals[-1]:.1f}x the last")
        tail = vals[len(vals) // 4:]
        print(f"  excluding the first quarter: min {min(tail):.4f} max {max(tail):.4f} "
              f"(spread {max(tail) - min(tail):.4f})")
        print()
        print("  first 8:", "  ".join(f"{v:.4f}" for v in vals[:8]))
        print("  last 8: ", "  ".join(f"{v:.4f}" for v in vals[-8:]))
    else:
        print("ATR series: EMPTY — the server returned no atr array.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
