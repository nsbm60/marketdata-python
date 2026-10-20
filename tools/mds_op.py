#!/usr/bin/env python3
"""
Call any MDS (or CalcServer) control op and print the reply.

There are a dozen one-off scripts in this directory that each discover a
service, open a DEALER socket and send one specific op. This is the generic
one: name the op, pass its arguments, see the raw reply.

Usage:
    PYTHONPATH=. python tools/mds_op.py get_option_snapshots --symbols MU260921P01015000
    PYTHONPATH=. python tools/mds_op.py get_option_snapshots --symbols MU260921P01015000,NVDA260921P00220000
    PYTHONPATH=. python tools/mds_op.py market_state --arg sec_type=STK
    PYTHONPATH=. python tools/mds_op.py --service calc debug_live_option_prices --symbols MU260921P01015000

Arguments:
    --symbols   comma-separated, sent as the "symbols" array
    --arg k=v   any other field; repeatable. Numbers and true/false are sent
                as their JSON types, everything else as a string.
"""
from __future__ import annotations

import argparse
import json
import sys

from control_client import ControlSocket, ControlTimeout
from discovery import ControlService, ServiceLocator

SERVICES = {
    "mds": ControlService.MARKET_DATA,
    "calc": ControlService.CALC,
    "ib": ControlService.IB,
}


def coerce(v: str):
    """Send numbers as numbers and booleans as booleans; everything else stays a string."""
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("op", help="control op name, e.g. get_option_snapshots")
    ap.add_argument("--service", default="mds", choices=sorted(SERVICES),
                    help="which service to call (default: mds)")
    ap.add_argument("--symbols", default="", help="comma-separated symbols")
    ap.add_argument("--arg", action="append", default=[], metavar="K=V",
                    help="additional request field; repeatable")
    ap.add_argument("--timeout", type=float, default=15.0, help="reply timeout seconds")
    ap.add_argument("--compact", action="store_true", help="one-line JSON instead of indented")
    args = ap.parse_args()

    try:
        endpoint = ServiceLocator.control(SERVICES[args.service], timeout_sec=10)
    except Exception as e:
        print(f"Discovery failed for {args.service}: {e}", file=sys.stderr)
        return 3

    router = endpoint.router
    if not router:
        print(f"{args.service} discovery returned no router endpoint: {endpoint}", file=sys.stderr)
        return 3
    print(f"{args.service} control router: {router}", file=sys.stderr)

    request = {"op": args.op}
    if args.symbols:
        request["symbols"] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    for pair in args.arg:
        if "=" not in pair:
            print(f"--arg must be k=v, got: {pair}", file=sys.stderr)
            return 2
        k, v = pair.split("=", 1)
        request[k] = coerce(v)

    print(f"request: {json.dumps(request)}", file=sys.stderr)

    sock = ControlSocket(router, int(args.timeout * 1000))
    try:
        parsed = sock.request(request)
    except ControlTimeout:
        print(f"no reply within {args.timeout}s", file=sys.stderr)
        return 1
    finally:
        sock.close()

    print(json.dumps(parsed, separators=(",", ":")) if args.compact
          else json.dumps(parsed, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
