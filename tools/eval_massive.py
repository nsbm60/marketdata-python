#!/usr/bin/env python3
"""
Pull NVDA option chain Greeks from Massive for a given expiry.
Prints a structured table for comparison against Alpaca and Fidelity.

Usage:
    export MASSIVE_API_KEY=your_key_here
    python tools/eval_massive.py NVDA 2026-05-15
    python tools/eval_massive.py NVDA 2026-05-15 --debug
"""

import argparse
import json
import os
import sys
from massive import RESTClient


def to_jsonable(obj):
    """Recursively convert an object graph to JSON-serializable primitives."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return repr(obj)


def fetch_chain(client, underlying: str, expiry: str):
    """Fetch all option contracts for an underlying at a specific expiry."""
    contracts = []
    for c in client.list_snapshot_options_chain(
        underlying,
        params={"expiration_date": expiry},
    ):
        contracts.append(c)
    return contracts


def format_row(c) -> str:
    """Format one contract's data as a table row."""
    details = c.details
    strike = details.strike_price
    contract_type = details.contract_type  # 'call' or 'put'

    g = c.greeks
    delta = g.delta if g else None
    gamma = g.gamma if g else None
    theta = g.theta if g else None
    vega = g.vega if g else None

    iv = c.implied_volatility
    oi = c.open_interest

    q = c.last_quote
    bid = q.bid if q else None
    ask = q.ask if q else None

    def fmt(v, width, decimals):
        if v is None:
            return "—".rjust(width)
        return f"{v:.{decimals}f}".rjust(width)

    def fmt_int(v, width):
        if v is None:
            return "—".rjust(width)
        return f"{v}".rjust(width)

    return (
        f"{contract_type[0].upper():2} "
        f"{fmt(strike, 8, 2)} "
        f"{fmt(bid, 8, 2)} "
        f"{fmt(ask, 8, 2)} "
        f"{fmt(iv, 8, 4)} "
        f"{fmt(delta, 9, 4)} "
        f"{fmt(gamma, 8, 4)} "
        f"{fmt(theta, 9, 4)} "
        f"{fmt(vega, 8, 4)} "
        f"{fmt_int(oi, 7)}"
    )


def pick_sample(contracts):
    """Pick a near-ATM call as a representative contract for raw dumping."""
    calls = sorted(
        (c for c in contracts if c.details.contract_type == "call"),
        key=lambda c: c.details.strike_price,
    )
    if not calls:
        return contracts[0] if contracts else None
    return calls[len(calls) // 2]


def main():
    parser = argparse.ArgumentParser(description="Massive option chain tester")
    parser.add_argument("underlying")
    parser.add_argument("expiry", help="YYYY-MM-DD")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dump full raw JSON for one near-ATM call before the table",
    )
    parser.add_argument(
        "--debug-all",
        action="store_true",
        help="Dump full raw JSON for every contract (verbose; pipe to a file)",
    )
    args = parser.parse_args()

    underlying = args.underlying.upper()
    expiry = args.expiry

    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        print("MASSIVE_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = RESTClient(api_key=api_key)

    print(f"Fetching {underlying} chain for expiry {expiry}...")
    contracts = fetch_chain(client, underlying, expiry)
    print(f"Got {len(contracts)} contracts\n")

    contracts.sort(key=lambda c: (c.details.contract_type, c.details.strike_price))

    if args.debug_all:
        print("--- Raw JSON for all contracts ---")
        print(json.dumps([to_jsonable(c) for c in contracts], indent=2, default=str))
        print()
    elif args.debug:
        sample = pick_sample(contracts)
        if sample is not None:
            d = sample.details
            print(f"--- Raw JSON for {d.contract_type} {d.strike_price} ---")
            print(json.dumps(to_jsonable(sample), indent=2, default=str))
            print()

    # Header
    print(
        "T   Strike      Bid      Ask       IV     Delta    Gamma     Theta     Vega      OI"
    )
    print("-" * 88)
    for c in contracts:
        print(format_row(c))


if __name__ == "__main__":
    main()
