#!/usr/bin/env python3
"""Hit the RAW Massive REST snapshot endpoint EXACTLY as MDS does — /v3/snapshot/options/{und}
with strike_price.gte/.lte + expiration_date.gte/.lte — and report greeks/OI fill.

Purpose: eval_massive.py uses the Massive SDK and gets full greeks+OI. MDS uses this raw REST
path and gets sparse greeks + 0 OI. Running this localizes the fault:
  - raw call returns full greeks+OI  => bug is in MDS's Scala parse/handling, not the request.
  - raw call returns sparse/none     => MDS's request (endpoint/params) differs from the SDK's.

Usage: python tools/eval_massive_raw.py META 2026-06-10 [minStrike] [maxStrike]
"""
import os
import sys
import httpx

key = os.environ.get("MASSIVE_API_KEY")
if not key:
    print("MASSIVE_API_KEY not set", file=sys.stderr)
    sys.exit(1)

und = sys.argv[1] if len(sys.argv) > 1 else "META"
exp = sys.argv[2] if len(sys.argv) > 2 else "2026-06-10"
lo  = sys.argv[3] if len(sys.argv) > 3 else "0"
hi  = sys.argv[4] if len(sys.argv) > 4 else "100000"

params = {
    "limit": "250", "sort": "ticker", "order": "asc",
    "strike_price.gte": lo, "strike_price.lte": hi,
    "expiration_date.gte": exp, "expiration_date.lte": exp,
    "apiKey": key,
}
url = f"https://api.massive.com/v3/snapshot/options/{und}"
print(f"GET {url}")
print("params (minus apiKey): "
      + ", ".join(f"{k}={v}" for k, v in params.items() if k != "apiKey"))

r = httpx.get(url, params=params, timeout=30.0)
print("status:", r.status_code)
results = r.json().get("results", []) or []
with_g  = sum(1 for c in results if (c.get("greeks") or {}).get("delta") is not None)
with_oi = sum(1 for c in results if c.get("open_interest") is not None)
print(f"results: {len(results)}")
print(f"with greeks(delta): {with_g}/{len(results)}    with open_interest: {with_oi}/{len(results)}")
if results:
    n = results[0]
    print("first node top-level keys:", sorted(n.keys()))
    print("first node greeks:", n.get("greeks"))
    print("first node open_interest:", n.get("open_interest"))
