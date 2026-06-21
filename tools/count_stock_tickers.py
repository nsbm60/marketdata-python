#!/usr/bin/env python3
"""Count active stock tickers from Massive's /v3/reference/tickers (market=stocks, active=true).

The endpoint is paginated (Polygon-style: results + next_url, no grand-total field), so this
pages through every next_url and sums len(results).

Key sourced exactly like the other tools here: the MASSIVE_API_KEY environment variable.

Usage: python tools/count_stock_tickers.py
"""
import os
import sys
import httpx

key = os.environ.get("MASSIVE_API_KEY")
if not key:
    print("MASSIVE_API_KEY not set", file=sys.stderr)
    sys.exit(1)

BASE = "https://api.massive.com"
url = f"{BASE}/v3/reference/tickers"
params = {"market": "stocks", "active": "true", "limit": "1000", "apiKey": key}

print(f"GET {url}")
print("params (minus apiKey): "
      + ", ".join(f"{k}={v}" for k, v in params.items() if k != "apiKey"))

total = 0
pages = 0
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
        print(f"page {pages}: {len(results)} (running total {total})")
        next_url = body.get("next_url")
        if not next_url:
            break

print(f"\nTOTAL active stock tickers: {total}  (over {pages} pages)")
