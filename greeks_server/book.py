"""The book: held option prices and the last result per contract (greeks-server-spec.md §3, §5.1).

Compute thread only. A contract enters the book on its first option quote or trade (or when the
cold path prices it for get_greeks) and stays — the book is never cleared. Each held quote and
trade is kept only if it is at least as new as the one held, so a price and its clock move together.

Each contract's last result is held too, keyed by its priced_as_of (the moment T it was priced as
of). A freshly computed result replaces the held one only if it is priced as of a later moment, or
the same moment but turns a failure into a success — that is the whole ordering rule, and it is
what makes an out-of-order or duplicate tick harmless.

A contract is dropped once its expiry date's close has passed (loop.py decides, from the time
event's ts); drop() forgets everything about it.
"""

from __future__ import annotations

import logging
from typing import Optional, TypeVar

from greeks_server.compute import Contract
from greeks_server.message import OptionGreeksMessage
from greeks_server.wire import Quote, Trade

log = logging.getLogger(__name__)


class Book:
    def __init__(self) -> None:
        self.contracts: dict[str, Contract] = {}
        self.by_underlying: dict[str, set[str]] = {}
        self.option_quotes: dict[str, Quote] = {}
        self.option_trades: dict[str, Trade] = {}
        self.results: dict[str, OptionGreeksMessage] = {}
        self._rejected: set[str] = set()

    def ensure(self, osi: str) -> Optional[Contract]:
        """The contract for ``osi``, registering it on first sight. None if not a standard OSI."""
        contract = self.contracts.get(osi)
        if contract is not None:
            return contract
        if osi in self._rejected:
            return None
        try:
            contract = Contract.parse(osi)
        except ValueError:
            self._rejected.add(osi)
            log.warning("not a standard OSI, not priced: %s", osi)
            return None
        self.contracts[contract.osi] = contract
        self.by_underlying.setdefault(contract.underlying, set()).add(contract.osi)
        return contract

    def on_option_quote(self, osi: str, quote: Quote) -> bool:
        return _newer(self.option_quotes, osi, quote)

    def on_option_trade(self, osi: str, trade: Trade) -> bool:
        return _newer(self.option_trades, osi, trade)

    def accept(self, msg: OptionGreeksMessage) -> bool:
        """Store ``msg`` as the contract's result and return True if it wins the ordering: a later
        priced_as_of, or the same one turning a failure into a success. Otherwise leave the held
        result and return False (the caller publishes only what was accepted)."""
        held = self.results.get(msg.osi)
        later = held is None or msg.priced_as_of > held.priced_as_of
        promotes = (
            held is not None
            and msg.priced_as_of == held.priced_as_of
            and held.reason is not None
            and msg.reason is None
        )
        if later or promotes:
            self.results[msg.osi] = msg
            return True
        return False

    def drop(self, osi: str) -> None:
        contract = self.contracts.pop(osi, None)
        self.option_quotes.pop(osi, None)
        self.option_trades.pop(osi, None)
        self.results.pop(osi, None)
        if contract is not None:
            peers = self.by_underlying.get(contract.underlying)
            if peers is not None:
                peers.discard(osi)
                if not peers:
                    del self.by_underlying[contract.underlying]

    def underlyings(self) -> set[str]:
        return set(self.by_underlying)

    def clear(self) -> None:
        """Forget every contract, held price and result — the session open is a restart. The
        rejected-OSI set is kept (a non-standard OSI stays non-standard across sessions)."""
        self.contracts.clear()
        self.by_underlying.clear()
        self.option_quotes.clear()
        self.option_trades.clear()
        self.results.clear()


_T = TypeVar("_T", Quote, Trade)


def _newer(store: dict[str, _T], key: str, value: _T) -> bool:
    """Keep ``value`` if it is at least as new as the one held. True if it was kept."""
    held = store.get(key)
    if held is not None and held.ts_ms > value.ts_ms:
        return False
    store[key] = value
    return True
