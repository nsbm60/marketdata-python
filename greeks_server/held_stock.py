"""The latest live stock the server holds per underlying (greeks-server-spec.md §3, §5.1).

The live bus carries live ticks only, so the server keeps just the newest live equity quote and the
newest live equity trade per underlying (each newer-only) — no history and no window. "The stock
now" is the mid of a valid two-sided quote, else the last trade (the same rule as everywhere). A
live option tick pairs with this current live stock; both are live, so the pair is coherent. All of
it is cleared at the session open (the open is a restart, §5.3).
"""

from __future__ import annotations

from typing import Optional

from greeks_server.price import SelectedPrice, select_spot
from greeks_server.wire import Quote, Trade

_LATEST = 2**62  # select_spot's "at or before" bound, admitting the newest held tick


class HeldStock:
    def __init__(self) -> None:
        self._quotes: dict[str, Quote] = {}
        self._trades: dict[str, Trade] = {}

    def on_quote(self, symbol: str, quote: Quote) -> None:
        held = self._quotes.get(symbol)
        if held is None or quote.ts_ms >= held.ts_ms:
            self._quotes[symbol] = quote

    def on_trade(self, symbol: str, trade: Trade) -> None:
        held = self._trades.get(symbol)
        if held is None or trade.ts_ms >= held.ts_ms:
            self._trades[symbol] = trade

    def price(self, symbol: str) -> Optional[SelectedPrice]:
        """The current live stock for ``symbol`` (mid first, else last), or None if none is held."""
        return select_spot(self._quotes.get(symbol), [self._trades.get(symbol)], _LATEST)

    def has(self, symbol: str) -> bool:
        return symbol in self._quotes or symbol in self._trades

    def clear(self) -> None:
        self._quotes.clear()
        self._trades.clear()
