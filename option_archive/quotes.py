"""Massive option NBBO quotes — sync httpx, no asyncio. Mirrors greeks massive_trades.

Endpoint: ``GET /v3/quotes/{optionsTicker}``, fully paginated via ``next_url`` (no
cap — the whole day or nothing, per the no-truncation rule). Each page GET goes
through ``with_retry`` so 429 / 5xx / transport errors are retried with backoff; the
quote phase pools these per-contract pulls. Feeds the quote phase in ``archive.py``.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional

import httpx

from greeks.occ import strip_massive_prefix, to_massive_ticker
from greeks.pull.massive_trades import MASSIVE_BASE, session_bounds_utc
from option_archive.ingest_day import NbboQuote
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
) -> list[NbboQuote]:
    """All NBBO quotes for one contract on ``session_date`` (America/New_York),
    sorted ascending by sip timestamp. Fully paginated; each page retried on
    429 / 5xx / transport."""
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
            for row in body.get("results") or []:
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
    return out
