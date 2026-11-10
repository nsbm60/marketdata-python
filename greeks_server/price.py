"""Price selection (greeks-server-spec.md §4). Pure.

The option price for greeks is the mid of a **two-sided quote only** (bid > 0, ask > 0, ask ≥ bid).
There is no trade fallback: greeks from a stale last trade describe a market that no longer exists,
and a contract with no current quote has no market to trade. The caller also requires the quote to
be *current* (timestamped at or after the session open) — that check lives in the caller, which
knows the session open; `quote_mid` is the pure two-sided-mid, time-unbounded.

The stock is chosen separately and may be a trade (a print is a real price at its instant):
`select_spot` is mid-first then the newest valid trade, used for the underlying (held live stock and
the raw-facts `compute_greeks` spot). The cold path's stock comes straight from an MDS print.
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


def quote_mid(quote: Optional[Quote]) -> Optional[SelectedPrice]:
    """The mid of a valid two-sided quote, else None. No time bound — the caller gates on time
    (current for the live server, at-or-before as_of for compute_greeks). No trade fallback."""
    if quote is not None:
        bid, ask = quote.bid, quote.ask
        if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
            return SelectedPrice((bid + ask) / 2.0, PriceKind.MID, quote.ts_ms)
    return None


def select_spot(
    quote: Optional[Quote], trades: Iterable[Optional[Trade]], at_or_before_ms: int
) -> Optional[SelectedPrice]:
    """The stock at or before T: a valid two-sided quote's mid first, else the newest valid trade.
    The stock may be a trade — unlike the option, a print is an acceptable spot."""
    mid = quote_mid(quote)
    if mid is not None and mid.ts_ms <= at_or_before_ms:
        return mid
    valid = [t for t in trades if t is not None and t.ts_ms <= at_or_before_ms and t.price > 0]
    if not valid:
        return None
    newest = max(valid, key=lambda t: t.ts_ms)
    return SelectedPrice(newest.price, PriceKind.LAST, newest.ts_ms)
