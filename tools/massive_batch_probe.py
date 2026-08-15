#!/usr/bin/env python3
"""Massive BATCHED snapshot probe — validates the unified endpoint before we rewrite the Scala.

Instead of fanning out one chain call per underlying (massive_concurrency_probe), this hits the
unified snapshot endpoint with up to 250 tickers per call, as Massive recommended:

    GET /v3/snapshot?ticker.any_of=O:AAPL260618C00190000,O:TSLA...&limit=250

It pulls the SAME contract set as the fan-out probe (same ClickHouse band plan), flattens it to
O:-prefixed tickers, batches by 250, and reports the latency distribution. The thing to confirm:
batched returns full greeks/IV/OI with NO 30s tail (Massive measured ~0.5s for 2,500 contracts
across 10 calls, vs 35–45s fanning out).

Usage (run alongside the fan-out probe with the same band/expiry for an apples-to-apples diff):
    python tools/massive_batch_probe.py --band-pct 15 --expiry-days 90 --levels 8 --repeat 5
"""

import argparse
import asyncio
import os
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from massive_concurrency_probe import (
    ch_connect, fetch_spots, build_fanout_plan, pct,
    latency_summary, slowest_requests, utc_now_iso,
)

BASE_URL = "https://api.massive.com"
SANE_IV_MAX = 3.0
DEFAULT_UNDERLYINGS = ["SPY", "AAPL", "NVDA", "MSFT", "AVGO", "META", "GOOGL", "AMZN", "TSLA", "SMH"]


def parse_args():
    p = argparse.ArgumentParser(description="Massive batched (ticker.any_of) snapshot probe")
    p.add_argument("--underlyings", default=",".join(DEFAULT_UNDERLYINGS))
    p.add_argument("--expiry-days", type=int, default=90)
    p.add_argument("--band-pct", type=float, default=15.0)
    p.add_argument("--max-per-call", type=int, default=200, help="band sizing for the CH plan (not the batch size)")
    p.add_argument("--batch-size", type=int, default=250, help="tickers per unified call (Massive max: 250)")
    p.add_argument("--levels", default="8", help="comma-separated concurrency levels for the batched calls")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--timeout", type=float, default=60.0)
    return p.parse_args()


def osi_to_ticker(und: str, expiry_iso: str, strike, right: str) -> str:
    """O:-prefixed Massive ticker from (underlying, ISO expiry, strike, C/P).
    OSI tail = YYMMDD + C/P + strike*1000 (8 digits)."""
    yymmdd = expiry_iso[2:4] + expiry_iso[5:7] + expiry_iso[8:10]  # 2026-06-22 -> 260622
    return f"O:{und}{yymmdd}{right}{int(round(float(strike) * 1000)):08d}"


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


async def fetch_batch(client, sem, tickers, key, idx):
    params = {"ticker.any_of": ",".join(tickers), "limit": "250", "apiKey": key}
    async with sem:
        t0 = time.perf_counter()
        started = utc_now_iso()
        try:
            resp = await client.get(f"{BASE_URL}/v3/snapshot", params=params)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            status = resp.status_code
            results = (resp.json().get("results") or []) if status == 200 else []
            req_set = set(tickers)
            # matched = how many returned contracts are actually ones we asked for. If the
            # endpoint were ignoring ticker.any_of (returning arbitrary limit=250), this is ~0.
            matched = sum(1 for c in results if c.get("ticker") in req_set)
            greeks_ok = sum(
                1 for c in results
                if (c.get("greeks") or {}).get("delta") is not None and c.get("implied_volatility") is not None
            )
            with_oi = sum(1 for c in results if c.get("open_interest") is not None)
            return {
                "status": status, "latency_ms": latency_ms, "started": started,
                "underlying": f"batch{idx}", "expiry": "", "requested": len(tickers),
                "got_n": len(results), "matched": matched, "greeks_ok": greeks_ok,
                "with_oi": with_oi, "error": None,
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status": f"EXC:{type(e).__name__}", "latency_ms": latency_ms, "started": started,
                "underlying": f"batch{idx}", "expiry": "", "requested": len(tickers),
                "got_n": 0, "matched": 0, "greeks_ok": 0, "with_oi": 0, "error": str(e)[:160],
            }


async def run_level(level, ticker_batches, key, timeout):
    sem = asyncio.Semaphore(level)
    limits = httpx.Limits(max_connections=level, max_keepalive_connections=level)
    async with httpx.AsyncClient(http2=False, timeout=timeout, limits=limits) as client:
        t0 = time.perf_counter()
        recs = await asyncio.gather(
            *[fetch_batch(client, sem, b, key, i) for i, b in enumerate(ticker_batches)],
            return_exceptions=True,
        )
        wall = time.perf_counter() - t0
    records = []
    for r in recs:
        if isinstance(r, Exception):
            records.append({"status": f"EXC:{type(r).__name__}", "latency_ms": 0.0, "started": "?",
                            "underlying": "batch?", "expiry": "", "requested": 0, "got_n": 0,
                            "matched": 0, "greeks_ok": 0, "with_oi": 0, "error": str(r)[:160]})
        else:
            records.append(r)
    return records, wall


def summarize(level, rep, records, wall, n_contracts):
    statuses = Counter(str(r["status"]) for r in records)
    errs = sum(1 for r in records if str(r["status"]).startswith("EXC"))
    got = sum(r["got_n"] for r in records)
    matched = sum(r.get("matched", 0) for r in records)
    greeks = sum(r["greeks_ok"] for r in records)
    oi = sum(r["with_oi"] for r in records)
    lats = [r["latency_ms"] for r in records if r["latency_ms"] > 0]
    status_str = " ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
    print(f"N={level:<3} rep{rep}  batches={len(records):<3} err={errs:<2} "
          f"p50={pct(lats,50):6.0f}ms p95={pct(lats,95):7.0f}ms wall={wall:6.2f}s "
          f"got={got}/{n_contracts} matched={matched}/{got} greeks={greeks}/{got} oi={oi}/{got}  [{status_str}]")
    return records


async def main_async():
    args = parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        print("MASSIVE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    underlyings = [u.strip().upper() for u in args.underlyings.split(",") if u.strip()]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    print(f"Massive BATCH probe — unified /v3/snapshot?ticker.any_of (<= {args.batch_size}/call), "
          f"http1.1, {len(underlyings)} underlyings, expiry today..+{args.expiry_days}d")
    print("Connecting to ClickHouse for contract universe...")
    ch = ch_connect()
    spots = fetch_spots(ch, underlyings) if args.band_pct > 0 else {}
    bands, n_groups, n_split, no_spot = build_fanout_plan(
        ch, underlyings, args.expiry_days, args.max_per_call, spots, args.band_pct)

    tickers = sorted({
        osi_to_ticker(b["underlying"], b["expiry"], strike, right)
        for b in bands for (strike, right) in b["expected"]
    })
    ticker_batches = list(chunks(tickers, args.batch_size))
    print(f"Plan: {len(tickers)} contracts -> {len(ticker_batches)} batched calls "
          f"(<= {args.batch_size}/call) vs ~{len(bands)} fan-out calls (~{max(1, len(bands)//max(1,len(ticker_batches)))}x fewer).")
    if not tickers:
        print("No contracts in ClickHouse for those underlyings/window.", file=sys.stderr)
        return

    # Prove which endpoint we're actually hitting (unified ticker.any_of, NOT /options/{underlying}).
    sample_url = str(httpx.Request("GET", f"{BASE_URL}/v3/snapshot",
        params={"ticker.any_of": ",".join(ticker_batches[0]), "limit": "250", "apiKey": key}).url)
    print("Endpoint in use (first call, key redacted):")
    print("  " + sample_url.replace(key, "***")[:220] + (" ..." if len(sample_url) > 220 else ""))

    print("-" * 110)
    all_records = []
    for level in levels:
        for rep in range(1, args.repeat + 1):
            records, wall = await run_level(level, ticker_batches, key, args.timeout)
            summarize(level, rep, records, wall, len(tickers))
            all_records.extend(records)
    print("-" * 110)
    latency_summary(all_records, "Massive (batched)")
    print("-" * 110)
    slowest_requests(all_records, "Massive (batched)")
    print("Compare wall / p99 / slowest against massive_concurrency_probe.py --mode fanout (same band/expiry):")
    print("  Batched should be ~sub-second with NO tail; fan-out carries the 30s stalls.")


if __name__ == "__main__":
    asyncio.run(main_async())
