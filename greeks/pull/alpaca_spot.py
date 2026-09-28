"""Alpaca SIP equity trades → as-of spot at option trade time.

**Adjustment policy (non-negotiable for validation):** prices must be
**raw / unadjusted**. Alpaca stock *bars* expose ``Adjustment.RAW``; stock
*trades* are unadjusted prints — we still import and assert ``Adjustment.RAW``
as the package constant so no caller silently picks ``Adjustment.ALL``
(see ``ml/etl/alpaca_bars_etl.py`` trap).

Feed: ``DataFeed.SIP`` only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest

from greeks.pull.massive_trades import OptionTradePrint

# Explicit — never rely on library default by omission.
REQUIRED_ADJUSTMENT = Adjustment.RAW
REQUIRED_FEED = DataFeed.SIP

_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class EquityTradePrint:
    symbol: str
    trade_ts: datetime  # UTC
    price: float
    size: float


@dataclass(frozen=True)
class SpotAtTrade:
    """Spot joined to one option trade (stored on the trade row only)."""

    option_symbol: str
    option_trade_ts: datetime
    spot: float
    spot_trade_ts: datetime
    underlying: str


def assert_raw_adjustment() -> Adjustment:
    """Runtime guard used by CLI and tests — must remain RAW."""
    if REQUIRED_ADJUSTMENT is not Adjustment.RAW:
        raise RuntimeError(
            f"alpaca_spot must use Adjustment.RAW, got {REQUIRED_ADJUSTMENT!r}"
        )
    return REQUIRED_ADJUSTMENT


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return ts.astimezone(timezone.utc)


def _trades_from_response(
    raw: object, symbol: str
) -> list[EquityTradePrint]:
    out: list[EquityTradePrint] = []
    data = getattr(raw, "data", None) or {}
    symbol_trades = data.get(symbol.upper()) or data.get(symbol) or []
    for tr in symbol_trades:
        ts = _ensure_utc(tr.timestamp)
        out.append(
            EquityTradePrint(
                symbol=symbol.upper(),
                trade_ts=ts,
                price=float(tr.price),
                size=float(tr.size),
            )
        )
    return out


def fetch_equity_trades(
    client: StockHistoricalDataClient,
    symbol: str,
    session_date: date,
    *,
    feed: DataFeed = REQUIRED_FEED,
    chunk_hours: int = 1,
    max_retries: int = 4,
) -> list[EquityTradePrint]:
    """SIP equity trades for one calendar day (America/New_York).

    Pulls in hour chunks with retries. Full-day single requests for liquid
    names (e.g. NVDA) paginate millions of prints and often hit socket
    read timeouts; chunking keeps each request bounded.
    """
    assert_raw_adjustment()
    if feed is not DataFeed.SIP:
        raise ValueError(f"validation requires DataFeed.SIP, got {feed!r}")
    if chunk_hours < 1:
        raise ValueError("chunk_hours must be >= 1")

    # Trades endpoint has no Adjustment field; RAW is enforced by using trades
    # (not Adjustment.ALL bars) and the module constant above.
    start_local = datetime(
        session_date.year, session_date.month, session_date.day, tzinfo=_ET
    )
    end_local = start_local + timedelta(days=1)
    sym = symbol.upper()
    out: list[EquityTradePrint] = []
    cursor = start_local
    while cursor < end_local:
        chunk_end = min(cursor + timedelta(hours=chunk_hours), end_local)
        last_err: Optional[BaseException] = None
        for attempt in range(max_retries):
            try:
                req = StockTradesRequest(
                    symbol_or_symbols=sym,
                    start=cursor,
                    end=chunk_end,
                    feed=feed,
                    limit=10000,  # API max page size — 10x fewer round-trips than the default
                )
                raw = client.get_stock_trades(req)
                out.extend(_trades_from_response(raw, sym))
                last_err = None
                break
            except Exception as e:  # network / timeout — retry chunk
                last_err = e
                if attempt + 1 >= max_retries:
                    break
                # simple linear backoff; no asyncio/threading
                time.sleep(1.5 * (attempt + 1))
        if last_err is not None:
            raise RuntimeError(
                f"Alpaca SIP trades failed for {sym} "
                f"[{cursor.isoformat()} .. {chunk_end.isoformat()}): {last_err}"
            ) from last_err
        cursor = chunk_end

    out.sort(key=lambda t: t.trade_ts)
    return out


def spot_asof(
    equity_trades: Sequence[EquityTradePrint],
    as_of_utc: datetime,
) -> Optional[EquityTradePrint]:
    """Last equity trade with ``trade_ts <= as_of_utc``."""
    as_of = _ensure_utc(as_of_utc)
    best: Optional[EquityTradePrint] = None
    for tr in equity_trades:
        if tr.trade_ts <= as_of:
            best = tr
        else:
            break
    return best


def attach_spot_to_option_trades(
    option_trades: Sequence[OptionTradePrint],
    equity_trades: Sequence[EquityTradePrint],
    underlying: str,
) -> list[tuple[OptionTradePrint, Optional[SpotAtTrade]]]:
    """Pair each option trade with as-of raw spot (None if no prior equity print)."""
    # equity_trades assumed sorted ascending
    eq = list(equity_trades)
    j = 0
    last: Optional[EquityTradePrint] = None
    out: list[tuple[OptionTradePrint, Optional[SpotAtTrade]]] = []
    for ot in sorted(option_trades, key=lambda t: t.trade_ts):
        ots = _ensure_utc(ot.trade_ts)
        while j < len(eq) and eq[j].trade_ts <= ots:
            last = eq[j]
            j += 1
        if last is None:
            out.append((ot, None))
        else:
            out.append(
                (
                    ot,
                    SpotAtTrade(
                        option_symbol=ot.symbol,
                        option_trade_ts=ots,
                        spot=last.price,
                        spot_trade_ts=last.trade_ts,
                        underlying=underlying.upper(),
                    ),
                )
            )
    return out


def fetch_equity_session_close_print(
    client: StockHistoricalDataClient,
    symbol: str,
    session_date: date,
    *,
    feed: DataFeed = REQUIRED_FEED,
) -> EquityTradePrint:
    """Last SIP print in the final hour of the ET session day.

    Cheap spot for moneyness filters / seed-only queue work. Full as-of join
    still uses :func:`fetch_equity_trades`.
    """
    assert_raw_adjustment()
    if feed is not DataFeed.SIP:
        raise ValueError(f"validation requires DataFeed.SIP, got {feed!r}")
    end_local = datetime(
        session_date.year, session_date.month, session_date.day, tzinfo=_ET
    ) + timedelta(days=1)
    start_local = end_local - timedelta(hours=1)
    sym = symbol.upper()
    last_err: Optional[BaseException] = None
    for attempt in range(4):
        try:
            req = StockTradesRequest(
                symbol_or_symbols=sym,
                start=start_local,
                end=end_local,
                feed=feed,
                limit=10000,  # API max page size — 10x fewer round-trips than the default
            )
            prints = _trades_from_response(client.get_stock_trades(req), sym)
            if not prints:
                raise RuntimeError(
                    f"no Alpaca SIP prints for {sym} in last hour of {session_date}"
                )
            prints.sort(key=lambda t: t.trade_ts)
            return prints[-1]
        except Exception as e:
            last_err = e
            if attempt + 1 >= 4:
                break
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"Alpaca close print failed for {sym} on {session_date}: {last_err}"
    ) from last_err


def make_stock_client(api_key: str, api_secret: str) -> StockHistoricalDataClient:
    if not api_key or not api_secret:
        raise ValueError("Alpaca API key and secret are required")
    return StockHistoricalDataClient(api_key, api_secret)
