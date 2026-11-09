"""Price selection (greeks-server-spec.md §4). Pure.

A set of greeks is priced as of one moment T. The price used for it is the newest usable one at or
before T — there is no lower bound and no session. T comes from the data (the tick's own timestamp,
the time event's ts, or a cold-path quote's timestamp); nothing here reads a clock.

For the option price (used for IV) and, with more than one candidate, for spot:
  1. A valid two-sided quote (bid > 0, ask > 0, ask >= bid) at or before T -> mid.
  2. Otherwise a last trade at or before T -> last. For spot there may be several trade
     candidates; the newest valid one at or before T is used.
  3. Otherwise nothing.

Mid comes first on purpose: a last print can be stale while the quote has moved. (CalcServer's
display rule is last first; it answers a different question.) The rule lives in Python only.
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


def _quote_mid(quote: Optional[Quote], at_or_before_ms: int) -> Optional[SelectedPrice]:
    if quote is not None and quote.ts_ms <= at_or_before_ms:
        bid, ask = quote.bid, quote.ask
        if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
            return SelectedPrice((bid + ask) / 2.0, PriceKind.MID, quote.ts_ms)
    return None


def select_price(
    quote: Optional[Quote], trade: Optional[Trade], at_or_before_ms: int
) -> Optional[SelectedPrice]:
    """The option price at or before T: a valid two-sided quote's mid, else the last trade."""
    mid = _quote_mid(quote, at_or_before_ms)
    if mid is not None:
        return mid
    if trade is not None and trade.ts_ms <= at_or_before_ms and trade.price > 0:
        return SelectedPrice(trade.price, PriceKind.LAST, trade.ts_ms)
    return None


def select_spot(
    quote: Optional[Quote], trades: Iterable[Optional[Trade]], at_or_before_ms: int
) -> Optional[SelectedPrice]:
    """Spot with more than one trade candidate: a valid quote mid at or before T first, else the
    newest valid trade at or before T."""
    mid = _quote_mid(quote, at_or_before_ms)
    if mid is not None:
        return mid
    valid = [t for t in trades if t is not None and t.ts_ms <= at_or_before_ms and t.price > 0]
    if not valid:
        return None
    newest = max(valid, key=lambda t: t.ts_ms)
    return SelectedPrice(newest.price, PriceKind.LAST, newest.ts_ms)
