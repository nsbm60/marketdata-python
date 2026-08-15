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
from datetime import datetime, timezone

from massive import RESTClient


# ---------- helpers --------------------------------------------------------

def ns_to_iso(ns):
    """Convert nanosecond epoch timestamp to ISO-8601 UTC. Returns None on failure."""
    if ns is None:
        return None
    try:
        # Massive/Polygon use ns; some fields are ms. Heuristic: > 1e15 → ns.
        seconds = ns / 1e9 if ns > 1e15 else ns / 1e3
        return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def get_attr_chain(obj, *names):
    """Return first non-None attribute from `names` on `obj`, or None."""
    if obj is None:
        return None
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def quote_timestamp(q):
    """Extract a timestamp from a quote object, trying common field names."""
    return get_attr_chain(q, "sip_timestamp", "last_updated", "timestamp", "t")


def dump_raw(obj):
    """Best-effort recursive serialization of a snapshot object for debugging."""
    def to_dict(o):
        if o is None or isinstance(o, (str, int, float, bool)):
            return o
        if isinstance(o, (list, tuple)):
            return [to_dict(x) for x in o]
        if isinstance(o, dict):
            return {k: to_dict(v) for k, v in o.items()}
        if hasattr(o, "__dict__"):
            return {k: to_dict(v) for k, v in vars(o).items() if not k.startswith("_")}
        return repr(o)
    return json.dumps(to_dict(obj), indent=2, default=str)


# ---------- core -----------------------------------------------------------

def fetch_chain(client, underlying: str, expiry: str):
    """Fetch all option contracts for an underlying at a specific expiry."""
    return list(client.list_snapshot_options_chain(
        underlying,
        params={"expiration_date": expiry},
    ))


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

    q = c.last_quote
    bid = getattr(q, "bid", None) if q else None
    ask = getattr(q, "ask", None) if q else None
    midpoint = getattr(q, "midpoint", None) if q else None
    if midpoint is None and bid and ask:
        midpoint = (bid + ask) / 2

    def fmt(v, width, decimals):
        if v is None:
            return "—".rjust(width)
        return f"{v:.{decimals}f}".rjust(width)

    return (
        f"{contract_type[0].upper():2} "
        f"{fmt(strike, 8, 2)} "
        f"{fmt(bid, 8, 2)} "
        f"{fmt(ask, 8, 2)} "
        f"{fmt(midpoint, 8, 2)} "
        f"{fmt(iv, 8, 4)} "
        f"{fmt(delta, 9, 4)} "
        f"{fmt(gamma, 8, 4)} "
        f"{fmt(theta, 9, 4)} "
        f"{fmt(vega, 8, 4)}"
    )


def summarize(contracts):
    """Print diagnostic summary: field coverage and quote freshness."""
    n = len(contracts)
    if n == 0:
        return

    def has_pos(q, attr):
        v = getattr(q, attr, None) if q else None
        return v is not None and v > 0

    have_quote_obj = sum(1 for c in contracts if c.last_quote is not None)
    have_bid = sum(1 for c in contracts if has_pos(c.last_quote, "bid"))
    have_ask = sum(1 for c in contracts if has_pos(c.last_quote, "ask"))
    have_both = sum(
        1 for c in contracts
        if has_pos(c.last_quote, "bid") and has_pos(c.last_quote, "ask")
    )
    have_iv = sum(1 for c in contracts if c.implied_volatility is not None)
    have_greeks = sum(1 for c in contracts if c.greeks is not None)
    have_trade = sum(1 for c in contracts if getattr(c, "last_trade", None) is not None)

    print(f"\n--- Field coverage ({n} contracts) ---")
    print(f"  last_quote object present : {have_quote_obj}/{n}")
    print(f"  bid > 0                   : {have_bid}/{n}")
    print(f"  ask > 0                   : {have_ask}/{n}")
    print(f"  bid AND ask > 0           : {have_both}/{n}")
    print(f"  implied_volatility set    : {have_iv}/{n}")
    print(f"  greeks set                : {have_greeks}/{n}")
    print(f"  last_trade present        : {have_trade}/{n}")

    # Quote freshness — distribution of timestamps
    quote_ts = [quote_timestamp(c.last_quote) for c in contracts]
    quote_ts = [t for t in quote_ts if t is not None]
    print(f"\n--- Timestamps ---")
    print(f"  current UTC               : {datetime.now(timezone.utc).isoformat()}")
    if quote_ts:
        print(f"  newest quote              : {ns_to_iso(max(quote_ts))}")
        print(f"  oldest quote              : {ns_to_iso(min(quote_ts))}")
    else:
        print(f"  no quote timestamps found in any contract")

    # If snapshot has its own updated_at / day field, show it
    sample = contracts[0]
    snap_updated = get_attr_chain(sample, "updated", "last_updated")
    if snap_updated:
        print(f"  snapshot 'updated'        : {ns_to_iso(snap_updated)}")


def pick_atm_sample(contracts):
    """Pick a representative near-the-money call for the debug dump."""
    calls = [c for c in contracts if c.details.contract_type == "call"
             and c.implied_volatility is not None]
    if not calls:
        calls = [c for c in contracts if c.details.contract_type == "call"]
    if not calls:
        return contracts[0] if contracts else None
    # Middle of the strike range is a reasonable proxy when we have no spot
    calls.sort(key=lambda c: c.details.strike_price)
    return calls[len(calls) // 2]


# ---------- main -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Massive option chain dumper")
    parser.add_argument("underlying")
    parser.add_argument("expiry", help="YYYY-MM-DD")
    parser.add_argument(
        "--debug", action="store_true",
        help="Dump raw JSON for one ATM-ish contract before the table",
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

    if args.debug:
        sample = pick_atm_sample(contracts)
        if sample is not None:
            print(f"--- Raw snapshot for {sample.details.contract_type} "
                  f"{sample.details.strike_price} ---")
            print(dump_raw(sample))
            print()

    print(
        "T   Strike      Bid      Ask      Mid       IV     Delta    Gamma     Theta     Vega"
    )
    print("-" * 88)
    for c in contracts:
        print(format_row(c))

    summarize(contracts)


if __name__ == "__main__":
    main()
