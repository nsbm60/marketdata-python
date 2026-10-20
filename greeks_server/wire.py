"""Decode the MDS tick payloads the server consumes (greeks-server-spec.md §3, §4).

Wire shapes (flat JSON, see the Scala producers):
  md.option.quote.*  {"type":"quote","symbol":<OSI>,"bid":…,"ask":…,"bidSize":…,"askSize":…,"timestamp":…}
  md.option.trade.*  {"type":"trade","symbol":<OSI>,"price":…,"size":…,"timestamp":…}
  md.equity.quote.*  {"type":"quote","symbol":<SYM>,"bid":…,"ask":…,…,"timestamp":…}
  md.equity.trade.*  {"type":"trade","symbol":<SYM>,"price":…,"size":…,"timestamp":…}

The timestamp arrives in two formats on the same topics: epoch millis as a string (Massive
WebSocket) and an ISO-8601 instant (REST baseline; Alpaca, with nanoseconds). Both are accepted,
as CalcServer's marketTimestampMillis does. A tick whose timestamp is missing or unparseable is
rejected (None) — it never gets the wall clock. A side may be NaN or 0 (one-sided market); that
is a legal quote, and price selection decides what it is worth.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True)
class Quote:
    bid: Optional[float]
    ask: Optional[float]
    ts_ms: int


@dataclass(frozen=True)
class Trade:
    price: float
    ts_ms: int


@dataclass(frozen=True)
class Tick:
    """One decoded tick: the symbol it is about (OSI or equity symbol) and its quote or trade."""

    symbol: str
    quote: Optional[Quote] = None
    trade: Optional[Trade] = None


# Epoch millis are 13 digits from 2001 to 2286. Anything else (seconds, nanos) is not millis.
_MS_MIN = 1_000_000_000_000
_MS_MAX = 9_999_999_999_999
# A tick stamped further ahead of the wall clock than this is rejected: kept, it would outrank
# every real tick (newer-only) and pin the price for the rest of the session.
MAX_FUTURE_MS = 60_000


def _ms(v: int) -> Optional[int]:
    return v if _MS_MIN <= v <= _MS_MAX else None


def parse_market_ts(raw: Any) -> Optional[int]:
    """Epoch millis from a tick timestamp: digits = epoch millis, else ISO-8601. None if neither."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return _ms(raw)
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    if s.isdigit():
        return _ms(int(s))
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return int(dt.timestamp() * 1000)


def _side(v: Any) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def decode_tick(payload: str, now_ms: int) -> Optional[Tick]:
    """Decode a quote or trade payload. None if it isn't a well-formed one.

    ``now_ms`` (wall clock) only bounds the timestamp: a tick more than MAX_FUTURE_MS ahead of
    it is rejected. The tick's own time is never replaced by the wall clock.
    """
    try:
        d = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(d, dict):
        return None
    sym = d.get("symbol")
    if not isinstance(sym, str) or not sym.strip():
        return None
    ts = parse_market_ts(d.get("timestamp"))
    if ts is None or ts > now_ms + MAX_FUTURE_MS:
        return None
    kind = d.get("type")
    if kind == "quote":
        return Tick(symbol=sym.strip().upper(), quote=Quote(_side(d.get("bid")), _side(d.get("ask")), ts))
    if kind == "trade":
        px = _side(d.get("price"))
        if px is None or px <= 0:
            return None
        return Tick(symbol=sym.strip().upper(), trade=Trade(px, ts))
    return None
