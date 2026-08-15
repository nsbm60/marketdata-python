#!/usr/bin/env python3
"""
Alpaca REST concurrency probe — a control for tools/massive_concurrency_probe.py.

Same machine, same network, same fanout (per-(underlying,expiry,strike-band) calls
sized from the ClickHouse contract universe), but it hits ALPACA's option-snapshot
REST instead of Massive. Run both DURING RTH to isolate vendor vs local network:

  - Alpaca clean while Massive stalls  => the stalls are Massive-specific (NOT this
                                          Mac, its egress, or the internet generally).
  - both stall at the same time        => shared cause = this machine's path/egress.

It reuses the Massive probe's ClickHouse band-building verbatim (same contract
universe, same band%, same per-call sizing) so identical flags produce comparable
plans, and reports the same per-level summary: error/timeout count, p50/p95 latency,
wall. The point is the timeout/latency distribution, not Alpaca data quality.

Usage (run alongside the Massive probe with the SAME flags):
    export ALPACA_API_KEY=...  ALPACA_API_SECRET=...
    python tools/alpaca_concurrency_probe.py --levels 8,16,32 --band-pct 15 --timeout 3 --repeat 2
    # compare against:
    python tools/massive_concurrency_probe.py --mode fanout --levels 8,16,32 --band-pct 15 --timeout 3 --repeat 2
"""

import argparse
import asyncio
import os
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

# Reuse the Massive probe's CH connection, spot fetch, band plan, and percentile helper
# (it lives next to this file; importing it has no side effects — its run is __main__-guarded).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from massive_concurrency_probe import (
    ch_connect, fetch_spots, build_fanout_plan, pct, fmt_strike, summarize_fanout,
    latency_summary, slowest_requests, utc_now_iso,
)

ALPACA_URL = "https://data.alpaca.markets/v1beta1"
SANE_IV_MAX = 3.0  # match the Massive probe's "usable greeks" definition


def osi_strike_right(osi: str):
    """(strike, 'C'/'P') from an OSI symbol's fixed tail: ...YYMMDD + C/P + strike*1000 (8 digits).
    Matches the (round(strike,3), right) tuples build_fanout_plan stores in band['expected']."""
    return round(int(osi[-8:]) / 1000.0, 3), osi[-9]


def parse_args():
    p = argparse.ArgumentParser(description="Alpaca REST concurrency probe (control for the Massive probe)")
    p.add_argument("--levels", default="1,2,4,8,16,32",
                   help="comma-separated concurrency levels (default: 1,2,4,8,16,32)")
    # Same default universe as massive_concurrency_probe.DEFAULT_UNDERLYINGS so the plans match.
    p.add_argument("--underlyings", default="SPY,AAPL,NVDA,MSFT,AVGO,META,GOOGL,AMZN,TSLA,SMH",
                   help="comma-separated underlyings")
    p.add_argument("--expiry-days", type=int, default=45,
                   help="expiration_date range = today .. today+N days (default: 45)")
    p.add_argument("--band-pct", type=float, default=15.0,
                   help="fetch only strikes within +/- this %% of spot (default 15; 0 = full chain)")
    p.add_argument("--max-per-call", type=int, default=200,
                   help="max contracts per call; bands split to stay under this (default 200)")
    p.add_argument("--timeout", type=float, default=3.0, help="per-request timeout (s, default 3)")
    p.add_argument("--repeat", type=int, default=2, help="repeat each level K times (default 2)")
    p.add_argument("--retries", type=int, default=0,
                   help="retry count per request (default 0 — measure RAW stall rate)")
    p.add_argument("--feed", default="opra", help="Alpaca options feed (default opra)")
    p.add_argument("--http2", action="store_true", help="use HTTP/2 (default HTTP/1.1)")
    return p.parse_args()


async def fetch_band(client, sem, band, headers, feed, retries):
    """One Alpaca option-chain snapshot call for a single (underlying, expiry) strike band."""
    url = f"{ALPACA_URL}/options/snapshots/{band['underlying']}"
    params = {
        "feed": feed,
        "limit": "1000",
        "strike_price_gte": fmt_strike(band["lo"]),
        "strike_price_lte": fmt_strike(band["hi"]),
        "expiration_date_gte": band["expiry"],
        "expiration_date_lte": band["expiry"],
    }
    attempt = 0
    while True:
        attempt += 1
        async with sem:
            t0 = time.perf_counter()
            started = utc_now_iso()
            try:
                resp = await client.get(url, params=params, headers=headers)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                status = resp.status_code
                got_set = set()
                greeks_ok = 0
                truncated = False
                paginated = False
                if status == 200:
                    try:
                        body = resp.json()
                        snaps = body.get("snapshots", {}) or {}
                        paginated = bool(body.get("next_page_token"))
                        for osi, snap in snaps.items():
                            try:
                                got_set.add(osi_strike_right(osi))
                            except Exception:
                                pass
                            g = snap.get("greeks") or {}
                            iv = snap.get("impliedVolatility")
                            if g.get("delta") is not None and iv is not None and 0 < iv < SANE_IV_MAX:
                                greeks_ok += 1
                    except Exception:
                        truncated = True  # body didn't parse
                rec = {
                    "underlying": band["underlying"], "expiry": band["expiry"], "status": status,
                    "latency_ms": latency_ms, "started": started, "truncated": truncated,
                    "got_n": len(got_set), "greeks_ok": greeks_ok, "expected_n": band["n"],
                    "missing": len(band["expected"] - got_set),
                    "extra": len(got_set - band["expected"]),
                    "paginated": paginated, "error": None,
                }
            except Exception as e:  # timeout, conn reset, protocol error, etc.
                latency_ms = (time.perf_counter() - t0) * 1000.0
                rec = {
                    "underlying": band["underlying"], "expiry": band["expiry"],
                    "status": f"EXC:{type(e).__name__}", "latency_ms": latency_ms,
                    "started": started,
                    "truncated": False, "got_n": 0, "greeks_ok": 0,
                    "expected_n": band["n"], "missing": 0, "extra": 0,
                    "paginated": False, "error": str(e)[:160],
                }

        retriable = rec["truncated"] or isinstance(rec["status"], str)
        if retriable and attempt <= retries:
            continue
        return rec


async def run_level(level, bands, headers, feed, retries, http2, timeout):
    sem = asyncio.Semaphore(level)
    limits = httpx.Limits(max_connections=level, max_keepalive_connections=level)
    async with httpx.AsyncClient(http2=http2, timeout=timeout, limits=limits) as client:
        t0 = time.perf_counter()
        recs = await asyncio.gather(
            *[fetch_band(client, sem, b, headers, feed, retries) for b in bands],
            return_exceptions=True,
        )
        wall = time.perf_counter() - t0
    records = []
    for r in recs:
        if isinstance(r, Exception):
            records.append({"underlying": "?", "expiry": "", "status": f"EXC:{type(r).__name__}",
                            "latency_ms": 0.0, "truncated": False, "got_n": 0, "greeks_ok": 0,
                            "expected_n": 0, "missing": 0, "extra": 0, "paginated": False,
                            "error": str(r)[:160]})
        else:
            records.append(r)
    return records, wall


async def main_async():
    args = parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        print("Set ALPACA_API_KEY and ALPACA_API_SECRET", file=sys.stderr)
        sys.exit(1)
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    http2 = bool(args.http2)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    underlyings = [u.strip().upper() for u in args.underlyings.split(",") if u.strip()]

    print(f"Alpaca concurrency probe (control for Massive) — http{'2' if http2 else '1.1'}, "
          f"feed={args.feed}, retries={args.retries}, {len(underlyings)} underlyings, "
          f"expiry today..+{args.expiry_days}d, timeout={args.timeout}s")
    print(f"underlyings: {','.join(underlyings)}")
    print("Connecting to ClickHouse for contract universe...")
    ch = ch_connect()
    spots = fetch_spots(ch, underlyings) if args.band_pct > 0 else {}
    bands, n_groups, n_split, no_spot = build_fanout_plan(
        ch, underlyings, args.expiry_days, args.max_per_call, spots, args.band_pct)
    total_expected = sum(b["n"] for b in bands)
    band_desc = f"+/-{args.band_pct:g}% of spot" if args.band_pct > 0 else "full chain"
    print(f"Band: {band_desc}"
          + (f"  (spots: {', '.join(f'{k}={v:g}' for k, v in sorted(spots.items()))})"
             if spots else ""))
    if no_spot:
        print(f"  WARNING: no spot for {sorted(no_spot)} — kept ALL strikes for these "
              f"(not silently dropped); band not applied.")
    print(f"Plan: {len(bands)} calls across {n_groups} (underlying,expiry) groups "
          f"({n_split} groups split to stay <= {args.max_per_call}/call), "
          f"{total_expected} contracts expected (in-band).")
    if not bands:
        print("No contracts in ClickHouse for those underlyings/window — nothing to probe.",
              file=sys.stderr)
        return

    # Same row + verdict format as massive_concurrency_probe.py --mode fanout: this reuses
    # its summarize_fanout so the two outputs cannot drift.
    print("-" * 110)
    summaries = []
    all_records = []
    for level in levels:
        for rep in range(1, args.repeat + 1):
            records, wall = await run_level(
                level, bands, headers, args.feed, args.retries, http2, args.timeout)
            all_records.extend(records)
            summaries.append(summarize_fanout(level, rep, records, wall))
    print("-" * 110)
    print("VERDICT")
    any_trunc = any(s["trunc"] > 0 for s in summaries)
    any_err = any(s["err"] > 0 for s in summaries)
    any_paged = any(s.get("paged", 0) > 0 for s in summaries)
    any_missing = any(s.get("missing", 0) > 0 for s in summaries)
    if not any_trunc and not any_err and not any_paged:
        print("  No truncation, no errors, no pagination at ANY concurrency level.")
        if any_missing:
            print("  NOTE: CH<->Alpaca universe mismatches present (see MISSING/EXTRA above).")
        else:
            print("  => completeness exact: Alpaca returned every in-band contract ClickHouse expected.")
    else:
        bad = sorted({s["level"] for s in summaries if s["trunc"] or s["err"] or s.get("paged", 0)})
        print(f"  Truncation/errors/pagination observed at levels: {bad}")
        print("  => check status codes above: 429/5xx = throttle; chunked/stream-reset = transport.")
    if summaries:
        best = max(summaries, key=lambda s: s.get("got", 0))
        got, gok = best.get("got", 0), best.get("greeks_ok", 0)
        if got:
            print(f"  In-band greeks coverage: {gok}/{got} = {100.0*gok/got:.1f}% "
                  f"(usable = non-null delta, 0 < IV < {SANE_IV_MAX}).")
    print("-" * 110)
    latency_summary(all_records, "Alpaca (fanout)")
    print("-" * 110)
    slowest_requests(all_records, "Alpaca (fanout)")
    print("Compare err/p95 against massive_concurrency_probe.py --mode fanout with the SAME flags:")
    print("  Alpaca clean + Massive stalls => Massive-specific.  Both stall => this Mac's network/egress.")


if __name__ == "__main__":
    asyncio.run(main_async())
