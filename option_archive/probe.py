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


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="option_archive probe",
                                 description="characterize a Massive outage")
    ap.add_argument("--contract")
    ap.add_argument("--date")
    ap.add_argument("--underlying")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = get_config()
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
