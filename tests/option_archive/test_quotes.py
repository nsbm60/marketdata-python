"""Massive NBBO quote parsing (the fetch pagination mirrors the proven trade path)."""

from __future__ import annotations

from datetime import date

from option_archive.ingest_day import NbboQuote
from option_archive.quotes import _parse_quote, fetch_option_quotes_day


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


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # 200 OK
        pass

    def json(self) -> dict:
        return self._payload


class _FakePagedClient:
    """Serves canned pages in order; records how many GETs it answered."""

    def __init__(self, pages: list[dict]) -> None:
        self._pages = list(pages)
        self.calls = 0

    def get(self, url: str, params: object = None) -> _FakeResp:
        self.calls += 1
        return _FakeResp(self._pages.pop(0))

    def close(self) -> None:
        pass


def test_fetch_counts_pages_and_raw_rows() -> None:
    """pages = GETs made; quotes_fetched = RAW vendor rows across all pages, counted
    before the session-window filter (decision A). Epoch-ns timestamps fall outside
    the 2022 session, so nothing is retained — proving the count is raw, not kept."""
    p1 = {"results": [{"sip_timestamp": "1"}, {"sip_timestamp": "2"}], "next_url": "https://x/next"}
    p2 = {"results": [{"sip_timestamp": "3"}]}  # no next_url -> last page
    client = _FakePagedClient([p1, p2])
    pull = fetch_option_quotes_day("k", "SPY220617C00380000", date(2022, 6, 13), client=client)  # type: ignore[arg-type]
    assert client.calls == 2
    assert pull.pages == 2
    assert pull.quotes_fetched == 3   # 2 + 1 raw rows, regardless of the filter
    assert pull.quotes == []          # all out-of-session: retained != fetched
