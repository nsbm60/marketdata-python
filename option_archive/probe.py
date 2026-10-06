"""`python -m option_archive probe` — characterize a Massive outage in one command.

When Massive misbehaves, this answers in one shot: is it general vs day-specific vs
ticker-specific, and which endpoint FAMILY is down. Each check is a single RAW request
— NO retries, raw truth not ridden-out truth — reporting status + latency + result
size on one plain line. Prints no API key and no full URLs.

Default (no flags): a known-liquid SPY 2022 contract-day across three families —
  * quotes at limit=10 AND at limit=50000. A backend that serves tiny reads (200)
    while 502ing full 50k pages is capacity-starved, not down — a different restart
    decision; the pair makes that visible.
  * the contracts reference (enumeration family), one request.
  * the trades day-file HEAD on S3 (flat-file family).

Flags isolate the dimension by varying one input:
  --underlying SYM + --date YYYY-MM-DD  pick a near-ATM contract from that day's chain
  --contract OSI                        pin an exact contract
  --limit N                             override the large quotes page size

Exit 0 iff every check ANSWERED (any HTTP/S3 status, even 4xx/5xx, is an answer);
nonzero if a check got no response at all — so it scripts as a pre-restart gate.

``--width-test`` is a separate mode — a quote-width benchmark for the heavy regime
(the pool-16 knee was set on the old ~1-page June data; this regime is ~50x the
payload). It resolves a real kept-contract subset for a fat day the same way the
drain does (download → parse → enumerate + band filter), caps it to a fixed set, and
pulls that identical set at pool widths 16/24/32/48, one line per width:
wall, throughput, per-call latency p50/p95, 429/5xx counts, and a network-wait vs
local-CPU split — the split says whether width can help (network-bound) or the GIL-
serialized parse/sort/merge already dominates (width can't). No retries (raw truth).
Unlike the outage checks this needs the real environment (ClickHouse, S3 + vendor
keys), so it runs where the drain runs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, timedelta
from typing import Any, Mapping, Optional

import httpx
from botocore.exceptions import BotoCoreError, ClientError

from greeks.occ import parse_occ, strip_massive_prefix, to_massive_ticker
from greeks.pull.massive_trades import MASSIVE_BASE
from option_archive.config import get_config
from option_archive.ingest_day import FLATFILES_BUCKET, TRADES_KEY_TEMPLATE, make_s3_client

log = logging.getLogger("option_archive.probe")

_DEFAULT_UNDERLYING = "SPY"
_DEFAULT_DATE = date(2022, 6, 13)                 # a liquid 2022 session, quotes exist
_FALLBACK_CONTRACT = "SPY220617C00380000"         # near-ATM SPY on the default day, if the chain can't be derived
_QUOTES_SMALL = 10
_QUOTES_LARGE = 50000
_TIMEOUT_S = 60.0

# --width-test defaults: a known ~1.5-page/contract fat day, a fixed subset so the
# runs are comparable and quick, and the pool widths to sweep.
_WIDTH_TEST_DATE = date(2022, 9, 29)
_WIDTH_TEST_CONTRACTS = 2000
_WIDTH_TEST_WIDTHS = (16, 24, 32, 48)


def _pick_contract(results: list[Any], day: date) -> Optional[str]:
    """Near-ATM call for the earliest expiry on/after ``day`` — a liquid pick without
    needing a spot (median strike of that expiry is an ATM proxy)."""
    parsed: list[tuple[date, float, str]] = []
    for row in results:
        if not isinstance(row, Mapping):
            continue
        ticker = row.get("ticker") or row.get("option_symbol")
        exp, strike = row.get("expiration_date"), row.get("strike_price")
        if not ticker or exp is None or strike is None:
            continue
        if not str(row.get("contract_type", "")).lower().startswith("c"):
            continue
        try:
            parsed.append((date.fromisoformat(str(exp)[:10]), float(strike), strip_massive_prefix(str(ticker))))
        except (ValueError, TypeError):
            continue
    if not parsed:
        return None
    near = min(e for e, _, _ in parsed)
    same = sorted((s, o) for e, s, o in parsed if e == near)
    return same[len(same) // 2][1]  # median strike's OSI


def _quotes_check(key: str, osi: Optional[str], day: date, limit: int) -> tuple[bool, str]:
    tag = f"quotes    limit={limit:<5}"
    if osi is None:
        return False, f"{tag} (no contract to probe — reference did not answer)  NOT-ANSWERED"
    massive = to_massive_ticker(strip_massive_prefix(osi))
    params: dict[str, Any] = {"timestamp": day.isoformat(), "limit": limit, "order": "asc",
                              "sort": "timestamp", "apiKey": key}
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=_TIMEOUT_S) as h:
            r = h.get(f"{MASSIVE_BASE}/v3/quotes/{massive}", params=params)
        ms = (time.monotonic() - t0) * 1000
        size = f"rows={len(r.json().get('results') or [])}" if r.status_code == 200 else "rows=-"
        return True, f"{tag} {osi}@{day}  status={r.status_code}  {ms:.0f}ms  {size}"
    except Exception as e:  # noqa: BLE001 — no response is the signal; type only, never str(e) (URL/key)
        ms = (time.monotonic() - t0) * 1000
        return False, f"{tag} {osi}@{day}  NO-RESPONSE ({type(e).__name__})  {ms:.0f}ms"


def _reference_check(key: str, underlying: str, day: date) -> tuple[bool, str, list[Any]]:
    params = {
        "underlying_ticker": underlying,
        "expiration_date.gte": day.isoformat(),
        "expiration_date.lte": (day + timedelta(days=90)).isoformat(),
        "limit": "1000", "sort": "expiration_date", "order": "asc",
        "expired": "true", "apiKey": key,
    }
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=_TIMEOUT_S) as h:
            r = h.get(f"{MASSIVE_BASE}/v3/reference/options/contracts", params=params)
        ms = (time.monotonic() - t0) * 1000
        results = list(r.json().get("results") or []) if r.status_code == 200 else []
        size = f"n={len(results)}" if r.status_code == 200 else "n=-"
        return True, f"contracts {underlying}@{day}  status={r.status_code}  {ms:.0f}ms  {size}", results
    except Exception as e:  # noqa: BLE001
        ms = (time.monotonic() - t0) * 1000
        return False, f"contracts {underlying}@{day}  NO-RESPONSE ({type(e).__name__})  {ms:.0f}ms", []


def _s3_head_check(s3: Optional[object], day: date) -> tuple[bool, str]:
    key_path = TRADES_KEY_TEMPLATE.format(y=day.year, m=day.month, d=day.day)
    if s3 is None:
        return False, f"trades-s3 {day}  NOT-CONFIGURED (no S3 keys in env)"
    t0 = time.monotonic()
    try:
        resp = s3.head_object(Bucket=FLATFILES_BUCKET, Key=key_path)  # type: ignore[attr-defined]
        ms = (time.monotonic() - t0) * 1000
        return True, f"trades-s3 {day}  status=200  {ms:.0f}ms  bytes={resp.get('ContentLength')}"
    except ClientError as e:  # S3 answered with an error status (404/403/5xx) — that IS an answer
        ms = (time.monotonic() - t0) * 1000
        code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") \
            or e.response.get("Error", {}).get("Code")
        return True, f"trades-s3 {day}  status={code}  {ms:.0f}ms  bytes=-"
    except (BotoCoreError, OSError) as e:
        ms = (time.monotonic() - t0) * 1000
        return False, f"trades-s3 {day}  NO-RESPONSE ({type(e).__name__})  {ms:.0f}ms"


# ---------------------------------------------------------------------------
# --width-test: quote-width benchmark for the heavy regime
# ---------------------------------------------------------------------------


def _parse_widths(s: str) -> list[int]:
    widths = [int(x) for x in s.split(",") if x.strip()]
    if not widths or any(w <= 0 for w in widths):
        raise ValueError(f"--widths must be positive ints, got {s!r}")
    return widths


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list (0.0 if empty)."""
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def _summarize(wall: float, pulls: list[Any]) -> dict[str, float]:
    """Aggregate one width's per-contract BenchPulls into the reported metrics. Pure
    (no I/O) so it unit-tests without a network. net_s/cpu_s are summed across threads;
    their split is the headline — a high cpu share means the GIL serializes the parse/
    sort/merge and more pool width cannot help."""
    pages = sum(b.pages for b in pulls)
    net = sum(b.network_s for b in pulls)
    cpu = sum(b.local_s for b in pulls)
    lats = sorted(s.latency_s for b in pulls for s in b.page_stats)
    n429 = sum(1 for b in pulls for s in b.page_stats if s.status == 429)
    n5xx = sum(1 for b in pulls for s in b.page_stats if 500 <= s.status < 600)
    busy = net + cpu
    return {
        "wall": wall,
        "contracts": float(len(pulls)),
        "pages": float(pages),
        "c_per_s": len(pulls) / wall if wall else 0.0,
        "pg_per_s": pages / wall if wall else 0.0,
        "p50_ms": _percentile(lats, 0.50) * 1000,
        "p95_ms": _percentile(lats, 0.95) * 1000,
        "n429": float(n429),
        "n5xx": float(n5xx),
        "net_s": net,
        "cpu_s": cpu,
        "net_pct": 100 * net / busy if busy else 0.0,
        "cpu_pct": 100 * cpu / busy if busy else 0.0,
    }


def _format_width_line(width: int, m: dict[str, float]) -> str:
    return (
        f"width={width:<3} wall={m['wall']:.1f}s  "
        f"thru={m['c_per_s']:.1f} c/s {m['pg_per_s']:.1f} pg/s  "
        f"lat p50={m['p50_ms']:.0f}ms p95={m['p95_ms']:.0f}ms  "
        f"429={m['n429']:.0f} 5xx={m['n5xx']:.0f}  "
        f"net={m['net_s']:.1f}s cpu={m['cpu_s']:.1f}s "
        f"(net {m['net_pct']:.0f}% / cpu {m['cpu_pct']:.0f}%)"
    )


def _resolve_keep_subset(cfg: Any, day: date, n_contracts: int) -> list[tuple[Any, Any]]:
    """Reproduce _process_day's front half (download → parse → enumerate + band
    filter) to get the day's REAL kept contracts, then cap to a fixed, deterministic
    subset (sorted by OSI, first N) so every width runs the identical set. Needs the
    real environment: discovery/ClickHouse, S3 flat-file creds, Massive + Alpaca keys."""
    from greeks.pull.alpaca_spot import make_stock_client
    from ml.shared.clickhouse import get_ch_client
    from option_archive import archive
    from option_archive.ingest_day import parse_trades
    from option_archive.reference import watchlist_underlyings

    ch = get_ch_client()
    alpaca = make_stock_client(cfg.api_keys.alpaca_api_key, cfg.api_keys.alpaca_api_secret)
    s3 = make_s3_client(
        os.environ.get("MASSIVE_S3_ACCESS_KEY", ""),
        os.environ.get("MASSIVE_S3_SECRET_KEY", ""),
        connect_timeout=cfg.s3.connect_timeout, read_timeout=cfg.s3.read_timeout,
    )
    unders = watchlist_underlyings(ch, table=cfg.tables.watchlist)
    spots = archive._load_spots(alpaca, unders, day, day)
    spots_for_day = {u: closes[day] for u, closes in spots.items() if day in closes}
    with archive._download_day(s3, cfg, day) as path:
        raw = path.read_bytes()
    by_symbol = parse_trades(raw, frozenset(unders))
    keep, _misses = archive._classify_day(
        by_symbol, spots_for_day, cfg, cfg.api_keys.massive_api_key, ch, day, {}
    )
    subset = sorted(keep.items())[:n_contracts]
    log.info("width-test %s: %d contracts (subset of %d kept)", day, len(subset), len(keep))
    return subset


def _width_test(cfg: Any, day: date, n_contracts: int, widths: list[int]) -> None:
    """Resolve the real kept subset once, then pull it at each pool width — identical
    set, identical work, only the width varies — and print one comparable line per
    width. No retries (raw truth)."""
    from concurrent.futures import ThreadPoolExecutor

    from option_archive.quotes import benchmark_quote_pull

    api_key = cfg.api_keys.massive_api_key
    items = _resolve_keep_subset(cfg, day, n_contracts)

    def _one(item: tuple[Any, Any]) -> Any:
        osi, prints = item
        return benchmark_quote_pull(api_key, str(osi), day, prints)

    for width in widths:
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=width) as pool:
            pulls = list(pool.map(_one, items))
        print(_format_width_line(width, _summarize(time.monotonic() - t0, pulls)))


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="option_archive probe",
                                 description="characterize a Massive outage")
    ap.add_argument("--contract")
    ap.add_argument("--date")
    ap.add_argument("--underlying")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--width-test", action="store_true",
                    help="quote-width benchmark: pull a fixed kept-contract subset at several pool widths")
    ap.add_argument("--contracts", type=int, default=_WIDTH_TEST_CONTRACTS,
                    help="(--width-test) size of the fixed contract subset")
    ap.add_argument("--widths", default=",".join(str(w) for w in _WIDTH_TEST_WIDTHS),
                    help="(--width-test) comma-separated pool widths to sweep")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = get_config()

    if args.width_test:
        day = date.fromisoformat(args.date) if args.date else _WIDTH_TEST_DATE
        _width_test(cfg, day, args.contracts, _parse_widths(args.widths))
        return

    key = cfg.api_keys.massive_api_key
    day = date.fromisoformat(args.date) if args.date else _DEFAULT_DATE
    underlying = (args.underlying or (parse_occ(args.contract).root if args.contract else _DEFAULT_UNDERLYING)).upper()
    large = args.limit or _QUOTES_LARGE
    try:
        s3: Optional[object] = make_s3_client(
            os.environ.get("MASSIVE_S3_ACCESS_KEY", ""),
            os.environ.get("MASSIVE_S3_SECRET_KEY", ""),
            connect_timeout=cfg.s3.connect_timeout, read_timeout=cfg.s3.read_timeout,
            max_attempts=1,  # raw truth — no boto3 retry
        )
    except ValueError:
        s3 = None

    # Reference first: its chain also derives the quotes contract (one request, reused).
    ref_ok, ref_line, results = _reference_check(key, underlying, day)
    if args.contract:
        contract: Optional[str] = strip_massive_prefix(args.contract)
    else:
        contract = _pick_contract(results, day)
        if contract is None and underlying == _DEFAULT_UNDERLYING and day == _DEFAULT_DATE:
            contract = _FALLBACK_CONTRACT  # keep the default baseline probing quotes even if reference is down

    qs_ok, qs_line = _quotes_check(key, contract, day, _QUOTES_SMALL)
    ql_ok, ql_line = _quotes_check(key, contract, day, large)
    s3_ok, s3_line = _s3_head_check(s3, day)

    for line in (qs_line, ql_line, ref_line, s3_line):
        print(line)
    sys.exit(0 if (qs_ok and ql_ok and ref_ok and s3_ok) else 1)
