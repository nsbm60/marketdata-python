"""Per-session market state and the dirty set (greeks-server-spec.md §3, §5.1). Compute thread only.

The contract set is whatever has ticked this session: a contract enters on its first quote or
trade. Each quote and trade is kept only if it is newer than the one held — the price and its
clock move together. Cleared at the session close.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from greeks_server.compute import Contract
from greeks_server.wire import Quote, Tick, Trade

log = logging.getLogger(__name__)


class Book:
    def __init__(self) -> None:
        self.contracts: dict[str, Contract] = {}
        self.by_underlying: dict[str, set[str]] = {}
        self.option_quotes: dict[str, Quote] = {}
        self.option_trades: dict[str, Trade] = {}
        self.equity_quotes: dict[str, Quote] = {}
        self.equity_trades: dict[str, Trade] = {}
        self.dirty: set[str] = set()
        self._rejected: set[str] = set()

    def on_option(self, tick: Tick) -> None:
        contract = self.contracts.get(tick.symbol)
        if contract is None:
            if tick.symbol in self._rejected:
                return
            try:
                contract = Contract.parse(tick.symbol)
            except ValueError:
                self._rejected.add(tick.symbol)
                log.warning("not a standard OSI, not priced: %s", tick.symbol)
                return
            self.contracts[contract.osi] = contract
            self.by_underlying.setdefault(contract.underlying, set()).add(contract.osi)
        osi = contract.osi
        changed = False
        if tick.quote is not None:
            changed = _newer(self.option_quotes, osi, tick.quote)
        if tick.trade is not None:
            changed = _newer(self.option_trades, osi, tick.trade) or changed
        if changed:
            self.dirty.add(osi)

    def on_equity(self, tick: Tick) -> None:
        changed = False
        if tick.quote is not None:
            changed = _newer(self.equity_quotes, tick.symbol, tick.quote)
        if tick.trade is not None:
            changed = _newer(self.equity_trades, tick.symbol, tick.trade) or changed
        if changed:
            self.dirty.update(self.by_underlying.get(tick.symbol, ()))

    def mark_all_dirty(self) -> None:
        self.dirty.update(self.contracts)

    def take_dirty(self) -> list[Contract]:
        out = [self.contracts[o] for o in self.dirty]
        self.dirty = set()
        return out


_T = TypeVar("_T", Quote, Trade)


def _newer(store: dict[str, _T], key: str, value: _T) -> bool:
    """Keep ``value`` if it is at least as new as the one held. True if it was kept."""
    held = store.get(key)
    if held is not None and held.ts_ms > value.ts_ms:
        return False
    store[key] = value
    return True
