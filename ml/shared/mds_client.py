"""
ml/shared/mds_client.py

MDS (Market Data Service) RPC client for bar data access.
Uses ZMQ DEALER/ROUTER pattern for JSON-RPC style requests.
"""

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

import zmq

NY = ZoneInfo("America/New_York")


@dataclass
class BarRecord:
    """1-minute bar data."""
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trade_count: int
    session: int  # 0=pre, 1=regular, 2=post


@dataclass
class SessionSummary:
    """Summary of a trading session."""
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


def get_bars(
    router_url: str,
    symbol: str,
    bar_date: Optional[date] = None,
    period: str = "1m",
    session: int = 1,
    timeout_ms: int = 5000,
) -> list[BarRecord]:
    """
    Fetch bars from MDS via ZMQ RPC.

    Args:
        router_url: MDS router endpoint e.g. "tcp://192.168.37.191:6007"
        symbol: Equity symbol (e.g., "NVDA")
        bar_date: Date to fetch bars for (default: today)
        session: Session filter (0=pre, 1=regular, 2=post)
        timeout_ms: Request timeout in milliseconds

    Returns:
        List of BarRecord objects in chronological order

    Raises:
        RuntimeError: If MDS unreachable or returns error
    """
    ctx = zmq.Context()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, timeout_ms)
    dealer.connect(router_url)

    try:
        request = {
            "op": "get_bars",
            "symbol": symbol.upper(),
            "period": period,
            "session": session,
        }
        if bar_date:
            request["date"] = bar_date.isoformat()

        dealer.send_string(json.dumps(request))

        if dealer.poll(timeout=timeout_ms):
            response = json.loads(dealer.recv_string())
            if not response.get("ok"):
                raise RuntimeError(
                    f"MDS error for get_bars({symbol}): "
                    f"{response.get('error', 'unknown')}"
                )

            data = response["data"]
            bars = []
            for bar in data.get("bars", []):
                # Parse ISO timestamp to datetime with NY timezone
                ts_str = bar["ts"]
                if ts_str.endswith("Z"):
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                else:
                    ts = datetime.fromisoformat(ts_str)
                ts = ts.astimezone(NY)

                bars.append(BarRecord(
                    ts=ts,
                    open=bar["open"],
                    high=bar["high"],
                    low=bar["low"],
                    close=bar["close"],
                    volume=bar["volume"],
                    vwap=bar["vwap"],
                    trade_count=bar["trade_count"],
                    session=bar["session"],
                ))
            return bars
        else:
            raise RuntimeError(f"MDS timeout for get_bars({symbol})")
    finally:
        dealer.close(linger=0)
        ctx.term()


def get_prior_session(
    router_url: str,
    symbol: str,
    timeout_ms: int = 5000,
) -> Optional[SessionSummary]:
    """
    Fetch prior session summary from MDS.

    Args:
        router_url: MDS router endpoint e.g. "tcp://192.168.37.191:6007"
        symbol: Equity symbol (e.g., "NVDA")
        timeout_ms: Request timeout in milliseconds

    Returns:
        SessionSummary or None if no prior session data

    Raises:
        RuntimeError: If MDS unreachable
    """
    ctx = zmq.Context()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, timeout_ms)
    dealer.connect(router_url)

    try:
        request = {
            "op": "get_prior_session",
            "symbol": symbol.upper(),
        }

        dealer.send_string(json.dumps(request))

        if dealer.poll(timeout=timeout_ms):
            response = json.loads(dealer.recv_string())
            if not response.get("ok"):
                # No prior session is not an error, just return None
                error = response.get("error", "")
                if "No prior session" in error:
                    return None
                raise RuntimeError(
                    f"MDS error for get_prior_session({symbol}): {error}"
                )

            data = response["data"]
            return SessionSummary(
                date=date.fromisoformat(data["date"]),
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                volume=data["volume"],
            )
        else:
            raise RuntimeError(f"MDS timeout for get_prior_session({symbol})")
    finally:
        dealer.close(linger=0)
        ctx.term()


def subscribe_with_backfill(
    router_url: str,
    symbol: str,
    data_type: str,
    timeframe: str = "5m",
    timeout_ms: int = 5000,
) -> Optional[dict]:
    """
    Call MDS subscribe_with_backfill RPC.

    Per the subscribe-with-backfill ADR, the caller MUST subscribe to the
    relevant PUB topics BEFORE calling this function — otherwise the gap
    guarantee is void.

    Args:
        router_url: MDS router endpoint e.g. "tcp://192.168.37.191:6007"
        symbol: Equity symbol (e.g., "NVDA")
        data_type: Data type (e.g., "indicators")
        timeframe: Timeframe (e.g., "5m", "1m")
        timeout_ms: Request timeout in milliseconds

    Returns:
        Full response dict on success (caller checks response["ok"]).
        None on timeout.

    Raises:
        RuntimeError: If MDS unreachable
    """
    ctx = zmq.Context()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, timeout_ms)
    dealer.connect(router_url)

    try:
        request = {
            "op": "subscribe_with_backfill",
            "data_type": data_type,
            "symbol": symbol.upper(),
            "timeframe": timeframe,
        }

        dealer.send_string(json.dumps(request))

        if dealer.poll(timeout=timeout_ms):
            response = json.loads(dealer.recv_string())
            return response
        else:
            return None
    finally:
        dealer.close(linger=0)
        ctx.term()
