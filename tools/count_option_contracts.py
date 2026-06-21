#!/usr/bin/env python3
"""All option contracts for an underlying from Massive's /v3/reference/options/contracts.

Pages every next_url and reports total contracts + distinct expiries (earliest/latest), to
confirm the endpoint returns the FULL universe (incl. quarterlies/LEAPS), not just near-dated.
This is the discovery source for the rewritten universe loader.

Key from MASSIVE_API_KEY (same as the other tools).

Usage: python tools/count_option_contracts.py [UNDERLYING]   # default NVDA
"""
import os
import sys
import httpx

key = os.environ.get("MASSIVE_API_KEY")
if not key:
    print("MASSIVE_API_KEY not set", file=sys.stderr)
    sys.exit(1)

und = (sys.argv[1] if len(sys.argv) > 1 else "NVDA").upper()
BASE = "https://api.massive.com"
url = f"{BASE}/v3/reference/options/contracts"
params = {
    "underlying_ticker": und,
    "expired": "false",            # currently-listed contracts only
    "limit": "1000",
    "sort": "expiration_date",
    "order": "asc",
    "apiKey": key,
}

print(f"GET {url}  underlying_ticker={und}")

total = 0
pages = 0
expiries = set()
with httpx.Client(timeout=30.0) as client:
    next_url = None
    while True:
        if next_url:
            # next_url carries the query but not the apiKey — re-append it (as MDS does).
            sep = "&" if "?" in next_url else "?"
            r = client.get(f"{next_url}{sep}apiKey={key}")
        else:
            r = client.get(url, params=params)
        r.raise_for_status()
        body = r.json()
        results = body.get("results", []) or []
        total += len(results)
        pages += 1
        for c in results:
            e = c.get("expiration_date")
            if e:
                expiries.add(e)
        print(f"page {pages}: {len(results)} (running total {total})")
        next_url = body.get("next_url")
        if not next_url:
            break

exp_sorted = sorted(expiries)
print(f"\n{und}: {total} contracts, {len(exp_sorted)} distinct expiries")
if exp_sorted:
    print(f"  earliest: {exp_sorted[0]}   latest: {exp_sorted[-1]}")
    print(f"  expiries: {', '.join(exp_sorted)}")
    # Show the shape of one contract so we can confirm the fields we'll persist.
print()
