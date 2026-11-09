"""Book: register on first sight, newer-only held prices, result ordering, drop, non-OSI reject."""

from __future__ import annotations

from greeks.domain import FailureReason, RowStatus
from greeks_server.book import Book
from greeks_server.message import OptionGreeksMessage
from greeks_server.price import PriceKind
from greeks_server.wire import Quote, Trade

A = "NVDA261120C00180000"
B = "NVDA261120P00170000"
M = "MU261120C00200000"


def _ok(osi: str, priced_as_of: int) -> OptionGreeksMessage:
    return OptionGreeksMessage(
        osi=osi, status=RowStatus.SUCCESS, priced_as_of=priced_as_of, methodology_version="v2",
        iv=0.4, delta=0.5, gamma=0.01, theta=-0.1, vega=0.2, rho=-0.01,
        price=9.0, price_kind=PriceKind.MID, price_ts=priced_as_of,
        spot=181.0, spot_kind=PriceKind.MID, spot_ts=priced_as_of,
        forward=181.0, discount=0.99, t_years=0.08,
    )


def _fail(osi: str, priced_as_of: int) -> OptionGreeksMessage:
    return OptionGreeksMessage(
        osi=osi, status=RowStatus.FAILURE, reason=FailureReason.MISSING_SPOT,
        priced_as_of=priced_as_of, methodology_version="v2",
    )


def test_ensure_registers_once_and_indexes_by_underlying() -> None:
    b = Book()
    assert b.ensure(A) is not None
    assert b.ensure(A) is b.ensure(A)       # same object, registered once
    b.ensure(M)
    assert set(b.contracts) == {A, M}
    assert b.by_underlying == {"NVDA": {A}, "MU": {M}}
    assert b.underlyings() == {"NVDA", "MU"}


def test_non_standard_osi_rejected_and_not_reparsed() -> None:
    b = Book()
    assert b.ensure("NVDA1261120C00180000") is None
    assert b.ensure("NVDA1261120C00180000") is None
    assert not b.contracts


def test_held_quote_is_newer_only() -> None:
    b = Book()
    b.ensure(A)
    assert b.on_option_quote(A, Quote(1.0, 1.1, 20)) is True
    assert b.on_option_quote(A, Quote(9.0, 9.1, 10)) is False   # older: ignored
    assert b.option_quotes[A] == Quote(1.0, 1.1, 20)
    assert b.on_option_trade(A, Trade(2.0, 30)) is True


def test_accept_takes_a_later_priced_as_of_only() -> None:
    b = Book()
    assert b.accept(_ok(A, 100)) is True
    assert b.accept(_ok(A, 90)) is False     # earlier: rejected
    assert b.accept(_ok(A, 100)) is False    # same time, success over success: no promotion
    assert b.accept(_ok(A, 110)) is True
    assert b.results[A].priced_as_of == 110


def test_equal_time_success_replaces_a_failure() -> None:
    b = Book()
    assert b.accept(_fail(A, 100)) is True
    assert b.accept(_ok(A, 100)) is True                 # same time, promotes failure -> success
    assert b.results[A].status is RowStatus.SUCCESS
    assert b.accept(_fail(A, 100)) is False              # does not demote at equal time


def test_drop_forgets_everything() -> None:
    b = Book()
    b.ensure(A)
    b.ensure(B)
    b.on_option_quote(A, Quote(1.0, 1.1, 5))
    b.accept(_ok(A, 5))
    b.drop(A)
    assert A not in b.contracts and A not in b.option_quotes and A not in b.results
    assert b.by_underlying == {"NVDA": {B}}
    b.drop(B)
    assert b.by_underlying == {} and b.underlyings() == set()
