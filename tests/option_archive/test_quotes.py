"""Massive NBBO quote parsing (the fetch pagination mirrors the proven trade path)."""

from __future__ import annotations

from option_archive.ingest_day import NbboQuote
from option_archive.quotes import _parse_quote


def test_parse_quote_maps_fields() -> None:
    q = _parse_quote({"sip_timestamp": "123456789", "bid_price": 1.5, "ask_price": 1.7,
                      "bid_size": 4, "ask_size": 5})
    assert isinstance(q, NbboQuote)
    assert q.sip_timestamp_ns == 123456789
    assert q.bid == 1.5 and q.ask == 1.7 and q.bid_size == 4 and q.ask_size == 5


def test_parse_quote_missing_sip_is_none() -> None:
    assert _parse_quote({"bid_price": 1.0}) is None


def test_parse_quote_absent_side_is_none() -> None:
    q = _parse_quote({"sip_timestamp": 10, "bid_price": 2.0})  # one-sided quote
    assert q is not None and q.bid == 2.0 and q.ask is None and q.ask_size is None
