#!/usr/bin/env python3
"""
option_slot.py

Ask CalcServer what it actually holds in liveOptionPrices for one or more OSI
symbols, via the debug_live_option_prices op.

This is the authoritative value — the same slot the option chain, the positions
report and the quote_option model all read. If a screen shows something else,
the screen is transforming it.

Also reports a plausibility check on theta. A per-share, per-day theta cannot
exceed roughly the option's own value divided by its days to expiry: the
contract cannot decay by more than it is worth. If the stored theta is far
larger than that bound, it is not a per-day number.

Usage:
    python tools/option_slot.py MU260904C01000000
    python tools/option_slot.py MU260904C01000000 NVDA260904C00220000
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from control_client import ControlTimeout, request as control_request
from discovery.service_locator import ServiceLocator


def dte_from_osi(osi: str) -> int | None:
    """OSI: <root><YYMMDD><C|P><strike*1000, 8 digits>."""
    for i, ch in enumerate(osi):
        if ch.isdigit():
            body = osi[i:]
            break
    else:
        return None
    if len(body) < 6:
        return None
    try:
        yy, mm, dd = int(body[0:2]), int(body[2:4]), int(body[4:6])
        return (date(2000 + yy, mm, dd) - date.today()).days
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+", help="OSI symbols")
    ap.add_argument("--timeout", type=int, default=15000)
    args = ap.parse_args()

    calc = ServiceLocator.wait_for_service(service_name=ServiceLocator.CALC, timeout_sec=30)

    try:
        reply = control_request(calc.router, {
            "op": "debug_live_option_prices",
            "symbols": [s.upper() for s in args.symbols],
        }, timeout_ms=args.timeout)
    except ControlTimeout:
        print("No reply from CalcServer.")
        return 1

    if not reply.get("ok", False):
        print(f"Request failed: {reply.get('error')}")
        return 1

    data = reply.get("data", {})
    if "data" in data and isinstance(data["data"], dict):
        data = data["data"]

    for slot in data.get("slots", []):
        osi = slot.get("osi")
        print(f"\n{osi}")
        if not slot.get("present"):
            print(f"  NOT PRESENT in liveOptionPrices   (tracked={slot.get('tracked')})")
            continue

        def g(k):
            v = slot.get(k)
            return f"{v:.6f}" if isinstance(v, (int, float)) else "—"

        print(f"  tracked={slot.get('tracked')}  "
              f"quoteTs={slot.get('quoteTs')} tradeTs={slot.get('tradeTs')} "
              f"capturedAt={slot.get('capturedAt')}")
        print(f"  last {g('last')}   bid {g('bid')}   ask {g('ask')}")
        print(f"  delta {g('delta')}  gamma {g('gamma')}  theta {g('theta')}  "
              f"vega {g('vega')}  iv {g('iv')}")

        theta = slot.get("theta")
        mid = None
        if isinstance(slot.get("bid"), (int, float)) and isinstance(slot.get("ask"), (int, float)):
            mid = (slot["bid"] + slot["ask"]) / 2
        elif isinstance(slot.get("last"), (int, float)):
            mid = slot["last"]

        dte = dte_from_osi(osi or "")
        if isinstance(theta, (int, float)) and mid is not None and dte is not None:
            print(f"\n  plausibility:  option value ~{mid:.4f}, {dte} days to expiry")
            if dte > 0:
                bound = mid / dte
                print(f"    a per-day theta cannot be much beyond  {-bound:.6f}")
                print(f"    stored theta is                        {theta:.6f}")
                if bound > 0 and abs(theta) > 5 * bound:
                    ratio = abs(theta) / bound
                    print(f"    -> {ratio:.0f}x the bound. Not a per-day number.")
                    print(f"       theta/365 = {theta / 365.0:.6f}  "
                          f"(annualised would explain a ~365x overshoot)")
                else:
                    print("    -> within a plausible range for per-day theta.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
