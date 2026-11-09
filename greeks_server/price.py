"""Price selection (greeks-server-spec.md §4). Pure.

A price is usable only if its timestamp is within the window [session open, price-time]:
  - price-time is as_of while in session, the session's close off-session;
  - at or after the session open, so a previous session's print is never used;
  - at or before price-time, so a print from after the time being priced is never used
    (the off-hours bug: NVDA's 20:00 after-hours print priced a 16:00 close).

For the option price used for IV, and the same bounds for spot:
  1. A valid two-sided quote (bid > 0, ask > 0, ask >= bid) in the window -> mid.
  2. Otherwise a last trade in the window -> last. For the spot there may be more than one trade
     candidate (the underlying's latest and its regular-session close); the newest valid one is used.
  3. Otherwise nothing.

Mid comes first on purpose: a session-fresh last can be hours old while the quote has moved.
(CalcServer's display rule is last first; it answers a different question.) The rule exists in
Python only — CalcServer sends raw facts and the server chooses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from greeks_server.wire import Quote, Trade


class PriceKind(str, Enum):
    MID = "mid"
    LAST = "last"


@dataclass(frozen=True)
class SelectedPrice:
    value: float
    kind: PriceKind
    ts_ms: int


def _in_window(ts_ms: int, open_ms: int, price_time_ms: int) -> bool:
    return open_ms <= ts_ms <= price_time_ms


def _quote_mid(quote: Optional[Quote], open_ms: int, price_time_ms: int) -> Optional[SelectedPrice]:
    if quote is not None and _in_window(quote.ts_ms, open_ms, price_time_ms):
        bid, ask = quote.bid, quote.ask
        if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
            return SelectedPrice((bid + ask) / 2.0, PriceKind.MID, quote.ts_ms)
    return None


def select_price(
    quote: Optional[Quote], trade: Optional[Trade], open_ms: int, price_time_ms: int
) -> Optional[SelectedPrice]:
    mid = _quote_mid(quote, open_ms, price_time_ms)
    if mid is not None:
        return mid
    if trade is not None and _in_window(trade.ts_ms, open_ms, price_time_ms) and trade.price > 0:
        return SelectedPrice(trade.price, PriceKind.LAST, trade.ts_ms)
    return None


def select_spot(
    quote: Optional[Quote], trades: Iterable[Optional[Trade]], open_ms: int, price_time_ms: int
) -> Optional[SelectedPrice]:
    """Spot with more than one trade candidate (the underlying's latest and its regular-session
    close): a valid in-window quote mid first, else the newest valid in-window trade."""
    mid = _quote_mid(quote, open_ms, price_time_ms)
    if mid is not None:
        return mid
    valid = [t for t in trades if t is not None and _in_window(t.ts_ms, open_ms, price_time_ms) and t.price > 0]
    if not valid:
        return None
    newest = max(valid, key=lambda t: t.ts_ms)
    return SelectedPrice(newest.price, PriceKind.LAST, newest.ts_ms)
