"""Book: newer-only updates, dirty marking, underlying fan-out, non-OSI drop."""

from __future__ import annotations

from greeks_server.book import Book
from greeks_server.wire import Quote, Tick, Trade

A = "NVDA261120C00180000"
B = "NVDA261120P00170000"
M = "MU261120C00200000"


def test_contracts_enter_on_first_tick_and_go_dirty() -> None:
    b = Book()
    b.on_option(Tick(A, quote=Quote(1.0, 1.1, 10)))
    b.on_option(Tick(M, trade=Trade(2.0, 10)))
    assert set(b.contracts) == {A, M}
    assert b.by_underlying == {"NVDA": {A}, "MU": {M}}
    assert {c.osi for c in b.take_dirty()} == {A, M}
    assert not b.dirty


def test_older_quote_is_ignored_and_not_dirty() -> None:
    b = Book()
    b.on_option(Tick(A, quote=Quote(1.0, 1.1, 20)))
    b.take_dirty()
    b.on_option(Tick(A, quote=Quote(9.0, 9.1, 10)))
    assert b.option_quotes[A] == Quote(1.0, 1.1, 20)
    assert not b.dirty


def test_underlying_tick_dirties_its_contracts_only() -> None:
    b = Book()
    for osi in (A, B, M):
        b.on_option(Tick(osi, trade=Trade(1.0, 5)))
    b.take_dirty()
    b.on_equity(Tick("NVDA", quote=Quote(180.0, 180.1, 7)))
    assert b.dirty == {A, B}
    b.on_equity(Tick("TSLA", trade=Trade(250.0, 7)))  # no contracts: stored, nothing dirty
    assert b.dirty == {A, B} and "TSLA" in b.equity_trades


def test_mark_all_dirty() -> None:
    b = Book()
    for osi in (A, M):
        b.on_option(Tick(osi, trade=Trade(1.0, 5)))
    b.take_dirty()
    b.mark_all_dirty()
    assert b.dirty == {A, M}


def test_non_standard_osi_dropped() -> None:
    b = Book()
    b.on_option(Tick("NVDA1261120C00180000", quote=Quote(1.0, 1.1, 1)))
    b.on_option(Tick("NVDA1261120C00180000", quote=Quote(1.0, 1.1, 2)))
    assert not b.contracts and not b.dirty
