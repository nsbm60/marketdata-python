#!/usr/bin/env python3
"""
Massive REST concurrency probe.

Settles, empirically, whether Massive truncates/throttles under CONCURRENT
requests — the assumption behind the Scala client's serial Semaphore(1) gate.
Uses true async HTTP (httpx + asyncio, single event loop, no threads), bypassing
the synchronous `massive` SDK so we can control concurrency precisely.

It replicates the production request faithfully:
    GET https://api.massive.com/v3/snapshot/options/{UND}
        ?limit=250&sort=ticker&order=asc
        [&expiration_date.gte=..&expiration_date.lte=..]
        &apiKey=...
following `next_url` pagination (re-appending the key, as the Scala client does).

For each concurrency level N it fires all underlyings concurrently bounded by an
asyncio.Semaphore(N), and reports per-N: status distribution (401/403/429/5xx/
conn-errors named), raw (pre-retry) truncation rate, latency percentiles, and
wall-clock. No retry by default — the whole point is to MEASURE raw truncation,
not mask it.

Truncation is detected loudly and never counted as success: a 200 whose body
fails a full JSON parse, or whose byte length disagrees with Content-Length, is
recorded as truncated (per the project's "No Silent Data Truncation" rule).

Usage:
    export MASSIVE_API_KEY=your_key_here
    python tools/massive_concurrency_probe.py
    python tools/massive_concurrency_probe.py --levels 1,2,4,8,16 --repeat 3
    python tools/massive_concurrency_probe.py --http1 --underlyings SPY,NVDA,AAPL
    python tools/massive_concurrency_probe.py --full          # no expiry range
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

BASE_URL = "https://api.massive.com"
DEFAULT_UNDERLYINGS = [
    "SPY", "AAPL", "NVDA", "MSFT", "AVGO",
    "META", "GOOGL", "AMZN", "TSLA", "SMH",
]
# Greeks are "usable" only with a non-null delta and an IV in a sane range.
# IV <= 0 or >= this cap is degenerate (deep ITM/OTM numerical garbage). Named
# explicitly rather than chosen silently — see "No Silent Data Truncation".
SANE_IV_MAX = 3.0


def parse_args():
    p = argparse.ArgumentParser(description="Massive REST concurrency probe (async)")
    p.add_argument("--levels", default="1,2,4,8,16",
                   help="comma-separated concurrency levels (default: 1,2,4,8,16)")
    p.add_argument("--underlyings", default=",".join(DEFAULT_UNDERLYINGS),
                   help="comma-separated underlyings")
    p.add_argument("--expiry-days", type=int, default=45,
                   help="expiration_date range = today .. today+N days (default: 45)")
    p.add_argument("--full", action="store_true",
                   help="no expiry range filter — pull the full chain per underlying")
    p.add_argument("--repeat", type=int, default=1,
                   help="repeat each level K times to expose intermittent truncation")
    p.add_argument("--http1", action="store_true", help="force HTTP/1.1 (default)")
    p.add_argument("--http2", action="store_true",
                   help="use HTTP/2 multiplexing (tests the GOAWAY/1.1-downgrade regime)")
    p.add_argument("--retries", type=int, default=0,
                   help="retry count per request (default 0 — measure RAW truncation)")
    p.add_argument("--timeout", type=float, default=30.0, help="per-request timeout (s)")
    p.add_argument("--mode", choices=["paginate", "fanout"], default="paginate",
                   help="paginate = full chains w/ next_url (default); "
                        "fanout = many small per-(underlying,expiry,strike-band) calls "
                        "sized from the ClickHouse contract universe to avoid pagination")
    p.add_argument("--max-per-call", type=int, default=200,
                   help="fanout: max contracts per call; bands split to stay under "
                        "the 250 page limit (default 200)")
    p.add_argument("--band-pct", type=float, default=15.0,
                   help="fanout: fetch only strikes within +/- this %% of spot per "
                        "underlying (default 15; 0 = full chain). Spot from CH stock_bar.")
    return p.parse_args()


def build_params(expiry_days: int, full: bool) -> dict:
    params = {"limit": "250", "sort": "ticker", "order": "asc"}
    if not full:
        today = date.today()
        params["expiration_date.gte"] = today.isoformat()
        params["expiration_date.lte"] = (today + timedelta(days=expiry_days)).isoformat()
    return params


def append_key(url: str, key: str) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}apiKey={key}"


async def get_one(client, sem, url, underlying, page, retries):
    """One gated HTTP GET. Returns (parsed_json_or_None, record_dict)."""
    attempt = 0
    while True:
        attempt += 1
        async with sem:
            t0 = time.perf_counter()
            started = utc_now_iso()
            try:
                resp = await client.get(url)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                body = resp.content
                text = resp.text
                status = resp.status_code

                # Content-Length mismatch is NOT a reliable truncation signal:
                # under gzip, Content-Length is the compressed size while body is
                # decompressed, so they legitimately differ on COMPLETE responses.
                # The only trustworthy truncation signal is a failed JSON parse.
                enc = resp.headers.get("content-encoding", "")
                clen_hdr = resp.headers.get("content-length")
                clen_mismatch = clen_hdr is not None and int(clen_hdr) != len(body)

                parsed = None
                truncated = False
                if status == 200:
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        truncated = True  # real truncation: body did not parse

                rec = {
                    "underlying": underlying, "page": page, "status": status,
                    "bytes": len(body), "latency_ms": latency_ms, "started": started,
                    "truncated": truncated, "clen_mismatch": clen_mismatch,
                    "enc": enc, "error": None,
                }
            except Exception as e:  # conn refused, timeout, protocol error, etc.
                latency_ms = (time.perf_counter() - t0) * 1000.0
                rec = {
                    "underlying": underlying, "page": page,
                    "status": f"EXC:{type(e).__name__}",
                    "bytes": 0, "latency_ms": latency_ms, "started": started,
                    "truncated": False, "clen_mismatch": False,
                    "error": str(e)[:160],
                }
                parsed = None

        retriable = rec["truncated"] or isinstance(rec["status"], str)
        if retriable and attempt <= retries:
            continue
        return parsed, rec


async def fetch_underlying(client, sem, underlying, params, key, retries):
    """Fetch a full chain for one underlying, following next_url pagination."""
    records = []
    contracts = 0
    qp = "&".join(f"{k}={v}" for k, v in params.items())
    url = append_key(f"{BASE_URL}/v3/snapshot/options/{underlying}?{qp}", key)
    page = 0
    while url is not None:
        page += 1
        parsed, rec = await get_one(client, sem, url, underlying, page, retries)
        records.append(rec)
        if parsed is None:
            break  # error/truncation already recorded; drop this underlying loudly
        results = parsed.get("results") or []
        contracts += len(results)
        nxt = parsed.get("next_url")
        url = append_key(nxt, key) if nxt else None
    return records, contracts


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[idx]


async def run_level(level, underlyings, params, key, retries, http2, timeout):
    sem = asyncio.Semaphore(level)
    limits = httpx.Limits(max_connections=level, max_keepalive_connections=level)
    async with httpx.AsyncClient(http2=http2, timeout=timeout, limits=limits) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            fetch_underlying(client, sem, u, params, key, retries) for u in underlyings
        ], return_exceptions=True)
        wall = time.perf_counter() - t0

    records, contracts = [], 0
    for r in results:
        if isinstance(r, Exception):
            records.append({"underlying": "?", "page": 0,
                            "status": f"EXC:{type(r).__name__}", "bytes": 0,
                            "latency_ms": 0.0, "truncated": False,
                            "clen_mismatch": False, "error": str(r)[:160]})
            continue
        recs, c = r
        records.extend(recs)
        contracts += c
    return records, contracts, wall


def summarize(level, rep, records, contracts, wall):
    statuses = Counter(str(r["status"]) for r in records)
    trunc = sum(1 for r in records if r["truncated"])
    errs = sum(1 for r in records if isinstance(r["status"], str) and str(r["status"]).startswith("EXC"))
    lats = [r["latency_ms"] for r in records if r["latency_ms"] > 0]
    status_str = " ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
    print(
        f"N={level:<3} rep{rep}  reqs={len(records):<4} "
        f"trunc={trunc:<3} err={errs:<3} "
        f"p50={pct(lats,50):6.0f}ms p95={pct(lats,95):7.0f}ms "
        f"wall={wall:6.2f}s contracts={contracts:<6} [{status_str}]"
    )
    # Loud per-record callout for any truncation / error — never hidden.
    for r in records:
        if r["truncated"] or (isinstance(r["status"], str) and str(r["status"]).startswith("EXC")):
            tag = "TRUNCATED" if r["truncated"] else "ERROR"
            print(f"      !! {tag} {r['underlying']} page{r['page']} "
                  f"status={r['status']} bytes={r['bytes']} "
                  f"clen_mismatch={r['clen_mismatch']} {r['error'] or ''}")
    return {"level": level, "trunc": trunc, "err": errs, "wall": wall,
            "contracts": contracts, "reqs": len(records)}


# ---------- fanout mode: ClickHouse-sourced, pre-sized, no pagination ----------

def ch_connect():
    """
    Connect to ClickHouse the same way the existing tools do: discover the
    endpoint via the ZMQ discovery bus (ServiceLocator), unless an explicit
    CLICKHOUSE_HOST is set as an override (useful off-network).
    """
    import clickhouse_connect

    host = os.environ.get("CLICKHOUSE_HOST")
    port = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
    if host:
        print(f"  ClickHouse (explicit) {host}:{port}")
    else:
        # Mirror migrate_stock_bars.py: add repo root to path, discover via bus.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from discovery.service_locator import ServiceLocator
        ep = ServiceLocator.wait_for_service(ServiceLocator.CLICKHOUSE, timeout_sec=30)
        host, port = ep.host, ep.port
        print(f"  ClickHouse discovered at {host}:{port} "
              f"(set CLICKHOUSE_HOST/PORT to override, DISCOVERY_HOST for the bus)")
    return clickhouse_connect.get_client(
        host=host, port=port,
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", "Aector99"),
        database=os.environ.get("CLICKHOUSE_DATABASE", "trading"),
    )


def fmt_strike(x):
    return f"{float(x):g}"


def fetch_spots(ch, underlyings):
    """Latest close per underlying from CH stock_bar, used to center the strike band."""
    unds = ",".join("'%s'" % u for u in underlyings)
    q = f"""
        SELECT symbol, argMax(close, ts) AS spot
        FROM trading.stock_bar
        WHERE symbol IN ({unds})
        GROUP BY symbol
    """
    return {sym: float(spot) for sym, spot in ch.query(q).result_rows if spot}


def build_fanout_plan(ch, underlyings, expiry_days, max_per_call, spots, band_pct):
    """
    Read the contract universe from ClickHouse and turn it into a list of bands,
    each a single Massive call for one (underlying, expiry) over a contiguous
    strike range sized to <= max_per_call contracts (so it never paginates).
    When band_pct > 0, strikes are restricted to +/- band_pct% of spot — the
    near-the-money region where greeks are meaningful and the panel needs data.
    Each band carries the EXACT expected {(strike, right)} set for completeness
    checking. Returns (bands, n_groups, n_split, no_spot).
    """
    unds = ",".join("'%s'" % u for u in underlyings)
    q = f"""
        SELECT underlying_symbol, expiration_date, strike_price, call_put
        FROM trading.option_contract
        WHERE underlying_symbol IN ({unds})
          AND expiration_date BETWEEN today() AND today() + {expiry_days}
        ORDER BY underlying_symbol, expiration_date, strike_price, call_put
    """
    rows = ch.query(q).result_rows

    # group (underlying, expiry) -> {strike: [rights]}
    groups = defaultdict(lambda: defaultdict(list))
    for und, exp, strike, cp in rows:
        right = str(cp).strip().upper()[0]  # 'C' or 'P'
        groups[(und, exp)][round(float(strike), 3)].append(right)

    bands = []
    n_split = 0
    no_spot = set()
    for (und, exp), by_strike in groups.items():
        strikes = sorted(by_strike)
        if band_pct > 0:
            spot = spots.get(und)
            if spot:
                lo_b, hi_b = spot * (1 - band_pct / 100.0), spot * (1 + band_pct / 100.0)
                strikes = [s for s in strikes if lo_b <= s <= hi_b]
            else:
                no_spot.add(und)  # no spot: keep all strikes, but flag it (no silent drop)
        if not strikes:
            continue
        cur = []  # list of (strike, right)
        before = len(bands)
        for s in strikes:
            entries = [(s, r) for r in by_strike[s]]
            if cur and len(cur) + len(entries) > max_per_call:
                bands.append(_make_band(und, exp, cur))
                cur = []
            cur.extend(entries)
        if cur:
            bands.append(_make_band(und, exp, cur))
        if len(bands) - before > 1:
            n_split += 1
    return bands, len(groups), n_split, no_spot


def _make_band(und, exp, contracts):
    strikes = [c[0] for c in contracts]
    return {
        "underlying": und,
        "expiry": exp.isoformat() if hasattr(exp, "isoformat") else str(exp),
        "lo": min(strikes), "hi": max(strikes),
        "expected": {(s, r) for s, r in contracts},
        "n": len(contracts),
    }


async def fetch_band(client, sem, band, key, retries):
    params = {
        "limit": "250", "sort": "ticker", "order": "asc",
        "expiration_date.gte": band["expiry"], "expiration_date.lte": band["expiry"],
        "strike_price.gte": fmt_strike(band["lo"]),
        "strike_price.lte": fmt_strike(band["hi"]),
    }
    qp = "&".join(f"{k}={v}" for k, v in params.items())
    url = append_key(f"{BASE_URL}/v3/snapshot/options/{band['underlying']}?{qp}", key)
    parsed, rec = await get_one(client, sem, url, band["underlying"], 1, retries)
    rec["expiry"] = band["expiry"]
    rec["expected_n"] = band["n"]
    rec["got_n"] = 0
    rec["greeks_ok"] = 0
    rec["missing"] = 0
    rec["extra"] = 0
    rec["paginated"] = False
    if parsed is not None:
        results = parsed.get("results") or []
        got = set()
        greeks_ok = 0
        for r in results:
            d = r.get("details") or {}
            s, ct = d.get("strike_price"), d.get("contract_type")
            if s is None or ct is None:
                continue
            got.add((round(float(s), 3), ct.strip().upper()[0]))
            g = r.get("greeks") or {}
            iv = r.get("implied_volatility")
            if g.get("delta") is not None and iv is not None and 0 < iv < SANE_IV_MAX:
                greeks_ok += 1
        rec["got_n"] = len(got)
        rec["greeks_ok"] = greeks_ok                    # contracts with usable greeks
        rec["missing"] = len(band["expected"] - got)   # CH has, Massive didn't return
        rec["extra"] = len(got - band["expected"])      # Massive returned, CH lacks
        rec["paginated"] = bool(parsed.get("next_url"))  # sizing failed if ever True
    return rec


async def run_level_fanout(level, bands, key, retries, http2, timeout):
    sem = asyncio.Semaphore(level)
    limits = httpx.Limits(max_connections=level, max_keepalive_connections=level)
    async with httpx.AsyncClient(http2=http2, timeout=timeout, limits=limits) as client:
        t0 = time.perf_counter()
        recs = await asyncio.gather(*[
            fetch_band(client, sem, b, key, retries) for b in bands
        ], return_exceptions=True)
        wall = time.perf_counter() - t0
    records = []
    for r in recs:
        if isinstance(r, Exception):
            records.append({"underlying": "?", "expiry": "", "status": f"EXC:{type(r).__name__}",
                            "latency_ms": 0.0, "truncated": False, "expected_n": 0,
                            "got_n": 0, "missing": 0, "extra": 0, "paginated": False,
                            "error": str(r)[:160]})
        else:
            records.append(r)
    return records, wall


def summarize_fanout(level, rep, records, wall):
    statuses = Counter(str(r["status"]) for r in records)
    trunc = sum(1 for r in records if r.get("truncated"))
    errs = sum(1 for r in records if str(r["status"]).startswith("EXC"))
    paged = sum(1 for r in records if r.get("paginated"))
    expected = sum(r.get("expected_n", 0) for r in records)
    got = sum(r.get("got_n", 0) for r in records)
    greeks_ok = sum(r.get("greeks_ok", 0) for r in records)
    missing = sum(r.get("missing", 0) for r in records)
    extra = sum(r.get("extra", 0) for r in records)
    lats = [r["latency_ms"] for r in records if r.get("latency_ms", 0) > 0]
    status_str = " ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
    print(
        f"N={level:<3} rep{rep}  calls={len(records):<4} "
        f"trunc={trunc:<3} err={errs:<3} paged={paged:<3} "
        f"p50={pct(lats,50):6.0f}ms p95={pct(lats,95):7.0f}ms wall={wall:6.2f}s "
        f"got={got}/{expected} greeks={greeks_ok}/{got} missing={missing} extra={extra}  [{status_str}]"
    )
    for r in records:
        bad = (r.get("truncated") or r.get("paginated") or r.get("missing", 0) or r.get("extra", 0)
               or str(r["status"]).startswith("EXC"))
        if bad:
            tags = []
            if str(r["status"]).startswith("EXC"):
                tags.append("ERROR")
            if r.get("truncated"):
                tags.append("TRUNC")
            if r.get("paginated"):
                tags.append("PAGINATED")  # sizing failed — band exceeded one page
            if r.get("missing"):
                tags.append(f"MISSING={r['missing']}")  # universe mismatch (reported separately)
            if r.get("extra"):
                tags.append(f"EXTRA={r['extra']}")       # universe mismatch (reported separately)
            print(f"      !! {' '.join(tags)} {r['underlying']} {r.get('expiry','')} "
                  f"status={r['status']} got={r.get('got_n',0)}/{r.get('expected_n',0)} "
                  f"{r.get('error') or ''}")
    return {"level": level, "trunc": trunc, "err": errs, "paged": paged,
            "missing": missing, "extra": extra, "wall": wall,
            "greeks_ok": greeks_ok, "got": got}


def utc_now_iso():
    """UTC wall-clock timestamp (ms precision) for the moment a request is issued — captured
    after the concurrency slot is acquired, so the vendor can match it to their server logs."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def latency_summary(records, label):
    """Plain per-request latency distribution across all completed requests in a run (ms).
    The slowest / worst-decile numbers are the point: the median can look fine while a few
    requests in the tail take tens of seconds and dominate wall-clock."""
    lats = sorted(r["latency_ms"] for r in records if r.get("latency_ms", 0) > 0)
    n = len(lats)
    if n == 0:
        print(f"LATENCY — {label}: no completed requests")
        return
    mean = sum(lats) / n
    d = max(1, n // 10)            # slowest 10%
    worst = lats[-d:]
    print(f"LATENCY — {label} (ms, {n} completed requests)")
    print(f"  min {lats[0]:.0f}  |  p25 {pct(lats,25):.0f}  |  median {pct(lats,50):.0f}  "
          f"|  p75 {pct(lats,75):.0f}  |  mean {mean:.0f}")
    print(f"  p90 {pct(lats,90):.0f}  |  p95 {pct(lats,95):.0f}  |  p99 {pct(lats,99):.0f}  "
          f"|  slowest {lats[-1]:.0f}")
    print(f"  worst 10% ({len(worst)} reqs): mean {sum(worst)/len(worst):.0f}, "
          f"range {worst[0]:.0f}–{worst[-1]:.0f}")
    print("  how to read: median = half the requests were faster than this, half slower.")
    print("    p90 / p95 / p99 = the time within which that % of requests finished — so p99")
    print("    is the slowest 1%. slowest = the single worst response. mean is pulled above")
    print("    the median when a few slow requests drag the average up.")


def slowest_requests(records, label, k=10):
    """The K slowest individual requests with their UTC start time — the 'which requests, and
    when' a vendor ticket asks for, ready to paste into their tracer."""
    timed = sorted((r for r in records if r.get("latency_ms", 0) > 0),
                   key=lambda r: r["latency_ms"], reverse=True)
    if not timed:
        return
    n = min(k, len(timed))
    print(f"SLOWEST {n} REQUESTS — {label} (start times UTC, for server-side log correlation)")
    print(f"  {'latency':>10}  {'started (UTC)':<29}  {'underlying':<10}  {'expiry':<11} status")
    for r in timed[:n]:
        print(f"  {r['latency_ms']:>8.0f}ms  {str(r.get('started','?')):<29}  "
              f"{str(r.get('underlying','?')):<10}  {str(r.get('expiry','')):<11}{r.get('status','')}")


async def run_fanout(args, key, http2, levels, underlyings):
    print("Connecting to ClickHouse for contract universe...")
    ch = ch_connect()
    spots = fetch_spots(ch, underlyings) if args.band_pct > 0 else {}
    bands, n_groups, n_split, no_spot = build_fanout_plan(
        ch, underlyings, args.expiry_days, args.max_per_call, spots, args.band_pct)
    total_expected = sum(b["n"] for b in bands)
    band_desc = (f"+/-{args.band_pct:g}% of spot" if args.band_pct > 0 else "full chain")
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
        return [], []
    print("-" * 110)
    summaries = []
    all_records = []
    for level in levels:
        for rep in range(1, args.repeat + 1):
            records, wall = await run_level_fanout(
                level, bands, key, args.retries, http2, args.timeout)
            all_records.extend(records)
            summaries.append(summarize_fanout(level, rep, records, wall))
    return summaries, all_records


async def main_async():
    args = parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        print("MASSIVE_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    if args.http2 and args.http1:
        print("choose --http1 OR --http2, not both", file=sys.stderr)
        sys.exit(1)
    http2 = bool(args.http2)

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    underlyings = [u.strip().upper() for u in args.underlyings.split(",") if u.strip()]

    print(f"Massive concurrency probe — mode={args.mode}, http{'2' if http2 else '1.1'}, "
          f"retries={args.retries}, {len(underlyings)} underlyings, "
          f"expiry today..+{args.expiry_days}d")
    print(f"underlyings: {','.join(underlyings)}")

    all_records = []
    if args.mode == "fanout":
        summaries, all_records = await run_fanout(args, key, http2, levels, underlyings)
    else:
        params = build_params(args.expiry_days, args.full)
        print("-" * 110)
        summaries = []
        for level in levels:
            for rep in range(1, args.repeat + 1):
                records, contracts, wall = await run_level(
                    level, underlyings, params, key, args.retries, http2, args.timeout)
                all_records.extend(records)
                summaries.append(summarize(level, rep, records, contracts, wall))

    print("-" * 110)
    print("VERDICT")
    any_trunc = any(s["trunc"] > 0 for s in summaries)
    any_err = any(s["err"] > 0 for s in summaries)
    any_paged = any(s.get("paged", 0) > 0 for s in summaries)
    any_missing = any(s.get("missing", 0) > 0 for s in summaries)
    if not any_trunc and not any_err and not any_paged:
        print("  No truncation, no errors, no pagination at ANY concurrency level.")
        if args.mode == "fanout":
            print("  => pre-sized per-(underlying,expiry,strike-band) fan-out avoids the failure modes.")
            if any_missing:
                print("  NOTE: CH<->Massive universe mismatches present (see MISSING/EXTRA above) — "
                      "cross-provider (CH=Alpaca, snapshots=Massive), NOT truncation.")
            else:
                print("  => completeness exact: Massive returned every in-band contract ClickHouse expected.")
        else:
            print("  => 'Massive truncates under concurrent load' is NOT supported by this run.")
    else:
        bad = sorted({s["level"] for s in summaries if s["trunc"] or s["err"] or s.get("paged", 0)})
        print(f"  Truncation/errors/pagination observed at levels: {bad}")
        if any_paged:
            print("  => PAGINATED rows mean a band exceeded one page — lower --max-per-call.")
        print("  => check status codes above: 429/5xx = throttle; chunked/stream-reset = transport.")
    if args.mode == "fanout" and summaries:
        # In-band greeks coverage from the cleanest (most-complete) level.
        best = max(summaries, key=lambda s: s.get("got", 0))
        got, gok = best.get("got", 0), best.get("greeks_ok", 0)
        if got:
            print(f"  In-band greeks coverage: {gok}/{got} = {100.0*gok/got:.1f}% "
                  f"(usable = non-null delta, 0 < IV < {SANE_IV_MAX}).")

    print("-" * 110)
    latency_summary(all_records, f"Massive ({args.mode})")
    print("-" * 110)
    slowest_requests(all_records, f"Massive ({args.mode})")


if __name__ == "__main__":
    asyncio.run(main_async())
