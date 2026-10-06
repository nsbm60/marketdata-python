"""Massive option NBBO quotes — sync httpx, no asyncio. Mirrors greeks massive_trades.

Endpoint: ``GET /v3/quotes/{optionsTicker}``, fully paginated via ``next_url`` (no
cap — the whole day or nothing, per the no-truncation rule). Each page GET goes
through ``with_retry`` so 429 / 5xx / transport errors are retried with backoff; the
quote phase pools these per-contract pulls. Feeds the quote phase in ``archive.py``.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import httpx

from greeks.occ import strip_massive_prefix, to_massive_ticker
from greeks.pull.massive_trades import MASSIVE_BASE, session_bounds_utc
from option_archive.ingest_day import FlatTradePrint, NbboQuote, attach_quotes
from option_archive.retry import with_retry


def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _parse_quote(row: Mapping[str, Any]) -> Optional[NbboQuote]:
    sip = row.get("sip_timestamp")
    if sip is None:
        return None
    try:
        sip_ns = int(sip)
    except (TypeError, ValueError):
        return None
    return NbboQuote(
        sip_timestamp_ns=sip_ns,
        bid=_opt_float(row.get("bid_price")),
        ask=_opt_float(row.get("ask_price")),
        bid_size=_opt_int(row.get("bid_size")),
        ask_size=_opt_int(row.get("ask_size")),
    )


class QuotePull(NamedTuple):
    """One contract-day's quote pull: the in-session quotes joined to prints, plus the
    RAW pull size — ``pages`` GET'd and ``quotes_fetched`` vendor rows served across
    all pages (before the session-window filter). The two sizes isolate vendor data
    volume (pages/contract, rows/page) from Massive service time (quote_seconds/page);
    the quote phase sums them across contracts into the ledger."""

    quotes: list[NbboQuote]
    pages: int
    quotes_fetched: int


class PageStat(NamedTuple):
    """One HTTP page attempt during a benchmark pull: how long the GET took, the
    status it answered (0 = transport error, no response), and the rows it carried."""

    latency_s: float
    status: int
    rows: int


class BenchPull(NamedTuple):
    """One contract's benchmark pull (NO retry). Splits wall into network_s (time
    inside the HTTP GETs) and local_s (parse + sip-sort + merge, the CPU after the
    bytes land) — the split that says whether more pool width can help at all — and
    keeps every page's PageStat for latency percentiles and 429/5xx counts."""

    pages: int          # successful (200) pages pulled
    quotes_fetched: int  # raw vendor rows across those pages
    network_s: float
    local_s: float
    page_stats: list[PageStat]  # every attempt, incl. the failing one that stopped the pull


def _get_json(http: httpx.Client, url: str, params: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    resp = http.get(url, params=dict(params)) if params is not None else http.get(url)
    resp.raise_for_status()  # 429/5xx -> HTTPStatusError, classified retryable by with_retry
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("Massive quotes JSON must be an object")
    return data


def fetch_option_quotes_day(
    api_key: str,
    osi: str,
    session_date: date,
    *,
    timeout_s: float = 60.0,
    client: Optional[httpx.Client] = None,
) -> QuotePull:
    """All NBBO quotes for one contract on ``session_date`` (America/New_York),
    sorted ascending by sip timestamp, with the raw pull size (pages, vendor rows).
    Fully paginated; each page retried on 429 / 5xx / transport."""
    if not api_key:
        raise ValueError("Massive API key is required")
    massive = to_massive_ticker(strip_massive_prefix(osi))
    start_utc, end_utc = session_bounds_utc(session_date)
    params: dict[str, Any] = {
        "timestamp": session_date.isoformat(), "limit": 50000,
        "order": "asc", "sort": "timestamp", "apiKey": api_key,
    }
    url = f"{MASSIVE_BASE}/v3/quotes/{massive}"
    own = client is None
    http = client or httpx.Client(timeout=timeout_s)
    out: list[NbboQuote] = []
    pages = 0
    quotes_fetched = 0
    try:
        nxt: Optional[str] = None
        while True:
            page_url = url if nxt is None else (nxt if "apiKey=" in nxt else f"{nxt}&apiKey={api_key}")
            page_params = params if nxt is None else None
            # with_retry calls this synchronously before the loop advances, so the
            # captured page_url/page_params are the current page's.
            body = with_retry(
                lambda: _get_json(http, page_url, page_params),
                what=f"massive quotes {osi}@{session_date}",
            )
            results = body.get("results") or []
            pages += 1
            quotes_fetched += len(results)  # raw vendor rows served (pre session filter)
            for row in results:
                if not isinstance(row, Mapping):
                    continue
                q = _parse_quote(row)
                if q is not None and start_utc <= q.quote_ts <= end_utc:
                    out.append(q)
            nxt = body.get("next_url")
            if not nxt:
                break
    finally:
        if own:
            http.close()
    out.sort(key=lambda q: q.sip_timestamp_ns)
    return QuotePull(out, pages, quotes_fetched)


def benchmark_quote_pull(
    api_key: str,
    osi: str,
    session_date: date,
    prints: Sequence[FlatTradePrint],
    *,
    timeout_s: float = 60.0,
    client: Optional[httpx.Client] = None,
) -> BenchPull:
    """Pull one contract's day of quotes with NO retry — raw truth for the width
    benchmark. Times each page's HTTP GET (network_s) and the JSON-decode + parse +
    sip-sort + merge that follows (local_s), and records every page's
    (latency, status, rows). On a non-200 (429 / 5xx) or a transport error the stat
    is recorded and pagination STOPS: a ridden-out storm would hide the very ceiling
    this benchmark is looking for. Reuses the production endpoint, params, and
    `_parse_quote`/`attach_quotes` so the measured work matches the real pull. Prints
    nothing; never logs the URL or key."""
    if not api_key:
        raise ValueError("Massive API key is required")
    massive = to_massive_ticker(strip_massive_prefix(osi))
    start_utc, end_utc = session_bounds_utc(session_date)
    params: dict[str, Any] = {
        "timestamp": session_date.isoformat(), "limit": 50000,
        "order": "asc", "sort": "timestamp", "apiKey": api_key,
    }
    url = f"{MASSIVE_BASE}/v3/quotes/{massive}"
    own = client is None
    http = client or httpx.Client(timeout=timeout_s)
    out: list[NbboQuote] = []
    pages = 0
    quotes_fetched = 0
    network_s = 0.0
    local_s = 0.0
    stats: list[PageStat] = []
    try:
        nxt: Optional[str] = None
        while True:
            page_url = url if nxt is None else (nxt if "apiKey=" in nxt else f"{nxt}&apiKey={api_key}")
            page_params = params if nxt is None else None
            t0 = time.monotonic()
            try:
                resp = (http.get(page_url, params=dict(page_params))
                        if page_params is not None else http.get(page_url))
            except Exception:  # noqa: BLE001 — no response is a data point; type only, never str (URL/key)
                stats.append(PageStat(latency_s=time.monotonic() - t0, status=0, rows=0))
                break
            dt = time.monotonic() - t0
            network_s += dt
            if resp.status_code != 200:  # raw truth: record the ceiling, do not ride it out
                stats.append(PageStat(latency_s=dt, status=resp.status_code, rows=0))
                break
            l0 = time.monotonic()
            body = resp.json()  # JSON decode is CPU, not network wait
            results = (body.get("results") or []) if isinstance(body, dict) else []
            pages += 1
            quotes_fetched += len(results)
            stats.append(PageStat(latency_s=dt, status=200, rows=len(results)))
            for row in results:
                if not isinstance(row, Mapping):
                    continue
                q = _parse_quote(row)
                if q is not None and start_utc <= q.quote_ts <= end_utc:
                    out.append(q)
            local_s += time.monotonic() - l0
            nxt = body.get("next_url") if isinstance(body, dict) else None
            if not nxt:
                break
    finally:
        if own:
            http.close()
    l0 = time.monotonic()
    out.sort(key=lambda q: q.sip_timestamp_ns)
    attach_quotes(prints, out)  # merge-walk as-of join; result discarded — we measure its cost
    local_s += time.monotonic() - l0
    return BenchPull(pages, quotes_fetched, network_s, local_s, stats)
