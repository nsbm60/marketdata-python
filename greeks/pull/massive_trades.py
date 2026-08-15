"""Massive option trades — sync httpx, no asyncio.

Endpoint: ``GET /v3/trades/{optionsTicker}`` (e.g. ``O:NVDA260527C00100000``).
Retries with exponential backoff on 429 / 5xx / transport stalls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timezone
from typing import Any, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import httpx

from greeks.occ import strip_massive_prefix, to_massive_ticker

MASSIVE_BASE = "https://api.massive.com"
_ET = ZoneInfo("America/New_York")
_DEFAULT_MAX_ATTEMPTS = 5
_DEFAULT_BACKOFF_S = 1.0


@dataclass(frozen=True)
class OptionTradePrint:
    """One OPRA option trade print (SIP clock)."""

    symbol: str  # bare OSI
    trade_ts: datetime  # UTC, from sip_timestamp (ns)
    price: float
    size: float
    exchange: Optional[int]
    conditions: tuple[int, ...]
    sequence_number: Optional[int]
    sip_timestamp_ns: int


def session_bounds_utc(session_date: date) -> tuple[datetime, datetime]:
    """RTH-ish day bounds in UTC for trade pull (00:00–24:00 America/New_York)."""
    start_local = datetime.combine(session_date, dtime(0, 0), tzinfo=_ET)
    end_local = datetime.combine(session_date, dtime(23, 59, 59, 999000), tzinfo=_ET)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def ns_to_utc(sip_timestamp_ns: int) -> datetime:
    return datetime.fromtimestamp(sip_timestamp_ns / 1e9, tz=timezone.utc)


def _parse_trade(row: Mapping[str, Any], osi: str) -> Optional[OptionTradePrint]:
    sip = row.get("sip_timestamp")
    if sip is None:
        return None
    try:
        sip_ns = int(sip)
        price = float(row["price"])
        size = float(row.get("size", 0))
    except (KeyError, TypeError, ValueError):
        return None
    cond_raw = row.get("conditions") or []
    conditions: tuple[int, ...]
    if isinstance(cond_raw, Sequence) and not isinstance(cond_raw, (str, bytes)):
        conditions = tuple(int(c) for c in cond_raw)
    else:
        conditions = ()
    exch = row.get("exchange")
    seq = row.get("sequence_number")
    return OptionTradePrint(
        symbol=osi,
        trade_ts=ns_to_utc(sip_ns),
        price=price,
        size=size,
        exchange=int(exch) if exch is not None else None,
        conditions=conditions,
        sequence_number=int(seq) if seq is not None else None,
        sip_timestamp_ns=sip_ns,
    )


def fetch_option_trades_day(
    api_key: str,
    options_ticker: str,
    session_date: date,
    *,
    timeout_s: float = 60.0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    backoff_s: float = _DEFAULT_BACKOFF_S,
    client: Optional[httpx.Client] = None,
) -> list[OptionTradePrint]:
    """All trades for one contract on ``session_date`` (America/New_York calendar)."""
    if not api_key:
        raise ValueError("Massive API key is required")
    osi = strip_massive_prefix(options_ticker)
    massive = to_massive_ticker(osi)
    start_utc, end_utc = session_bounds_utc(session_date)
    # Massive accepts YYYY-MM-DD or ns; date form is enough for a session day.
    params: dict[str, str | int] = {
        "timestamp": session_date.isoformat(),
        "limit": 50000,
        "order": "asc",
        "sort": "timestamp",
        "apiKey": api_key,
    }
    url: Optional[str] = f"{MASSIVE_BASE}/v3/trades/{massive}"
    own = client is None
    http = client or httpx.Client(timeout=timeout_s)
    out: list[OptionTradePrint] = []
    try:
        next_url: Optional[str] = None
        while True:
            body = _get_with_retry(
                http,
                url if next_url is None else next_url,
                params=params if next_url is None else None,
                api_key=api_key,
                max_attempts=max_attempts,
                backoff_s=backoff_s,
            )
            results = body.get("results") or []
            if not isinstance(results, list):
                raise ValueError("Massive trades response missing results list")
            for row in results:
                if not isinstance(row, Mapping):
                    continue
                tr = _parse_trade(row, osi)
                if tr is None:
                    continue
                # Keep only prints whose SIP clock falls on the session calendar day in ET.
                if start_utc <= tr.trade_ts <= end_utc:
                    out.append(tr)
            next_url = body.get("next_url")
            if not next_url:
                break
            url = None
            params = {}
    finally:
        if own:
            http.close()
    out.sort(key=lambda t: (t.sip_timestamp_ns, t.sequence_number or 0))
    return out


def _get_with_retry(
    http: httpx.Client,
    url: Optional[str],
    *,
    params: Optional[Mapping[str, str | int]],
    api_key: str,
    max_attempts: int,
    backoff_s: float,
) -> dict[str, Any]:
    if not url:
        raise ValueError("url required")
    last_err: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            if params is not None and "apiKey" in dict(params):
                resp = http.get(url, params=dict(params))
            elif "?" in url:
                # next_url — re-append apiKey like other Massive tools
                sep = "&" if "apiKey=" not in url else None
                get_url = f"{url}&apiKey={api_key}" if sep == "&" else url
                if "apiKey=" not in get_url:
                    get_url = f"{url}&apiKey={api_key}"
                resp = http.get(get_url)
            else:
                resp = http.get(url, params={**(params or {}), "apiKey": api_key})
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = httpx.HTTPStatusError(
                    f"retryable status {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
                time.sleep(backoff_s * (2**attempt))
                continue
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("Massive trades JSON must be an object")
            return data
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last_err = e
            time.sleep(backoff_s * (2**attempt))
    assert last_err is not None
    raise last_err
