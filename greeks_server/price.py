"""Price selection (greeks-server-spec.md §4). Pure.

For the option price used for IV, and the same rule for spot:
  1. A valid two-sided quote (bid > 0, ask > 0, ask >= bid) timestamped this session -> mid.
  2. Otherwise a last trade from this session -> last.
  3. Otherwise nothing.

"This session" means at or after the session open (09:30 ET), the same boundary as CalcServer's
sessionLast/sessionPrice. Mid comes first on purpose: a session-fresh last can be hours old while
the quote has moved. (CalcServer's display rule is last first; it answers a different question.)

The rule exists in two languages — CalcServer builds compute_greeks requests with it at cutover —
and both test the same golden cases (tests/greeks_server/price_golden.json).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from greeks_server.wire import Quote, Trade


class PriceKind(str, Enum):
    MID = "mid"
    LAST = "last"


@dataclass(frozen=True)
class SelectedPrice:
    value: float
    kind: PriceKind
    ts_ms: int


def select_price(
    quote: Optional[Quote], trade: Optional[Trade], session_open_ms: int
) -> Optional[SelectedPrice]:
    if quote is not None and quote.ts_ms >= session_open_ms:
        bid, ask = quote.bid, quote.ask
        if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
            return SelectedPrice((bid + ask) / 2.0, PriceKind.MID, quote.ts_ms)
    if trade is not None and trade.ts_ms >= session_open_ms and trade.price > 0:
        return SelectedPrice(trade.price, PriceKind.LAST, trade.ts_ms)
    return None
