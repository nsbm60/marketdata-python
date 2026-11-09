"""Rolling stock-price history per underlying (greeks-server-spec.md §3, §5.1). Compute thread only.

The server answers "the stock price at T" from here, not from a price a client relayed. Every
equity quote (its mid, when two-sided) and every equity trade (its last) goes in as one (ts, price)
entry keyed by its own timestamp — one merged series per underlying, newest at or before T wins.
Entries older than one hour behind the newest one held for that underlying are dropped (the window
is measured against the data's own timestamps, never a wall clock). An underlying with no entry at
or before T is a miss: the caller falls back to the cold path.

A stock tick only updates the history; it never triggers repricing (see loop.py for why).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from greeks_server.price import PriceKind, SelectedPrice

HISTORY_MS = 60 * 60 * 1000  # one hour


class StockHistory:
    def __init__(self) -> None:
        # Per underlying, (ts_ms, price, kind) in arrival order; pruned to the last hour.
        self._series: dict[str, deque[tuple[int, float, PriceKind]]] = {}

    def add_quote(self, symbol: str, ts_ms: int, bid: Optional[float], ask: Optional[float]) -> None:
        if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
            self._add(symbol, ts_ms, (bid + ask) / 2.0, PriceKind.MID)

    def add_trade(self, symbol: str, ts_ms: int, price: float) -> None:
        if price > 0:
            self._add(symbol, ts_ms, price, PriceKind.LAST)

    def _add(self, symbol: str, ts_ms: int, price: float, kind: PriceKind) -> None:
        series = self._series.get(symbol)
        if series is None:
            series = deque()
            self._series[symbol] = series
        series.append((ts_ms, price, kind))
        newest = max(e[0] for e in series)
        cutoff = newest - HISTORY_MS
        while series and series[0][0] < cutoff:
            series.popleft()

    def at(self, symbol: str, t_ms: int) -> Optional[SelectedPrice]:
        """The newest retained price for ``symbol`` at or before ``t_ms``; None if there is none."""
        series = self._series.get(symbol)
        if not series:
            return None
        best: Optional[tuple[int, float, PriceKind]] = None
        for ts, price, kind in series:
            if ts <= t_ms and (best is None or ts > best[0]):
                best = (ts, price, kind)
        if best is None:
            return None
        return SelectedPrice(best[1], best[2], best[0])

    def drop_symbols(self, keep: set[str]) -> None:
        """Forget underlyings not in ``keep`` (a contract left the book)."""
        for sym in [s for s in self._series if s not in keep]:
            del self._series[sym]
