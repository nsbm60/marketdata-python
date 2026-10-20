"""Tick decoding: both timestamp formats, one-sided quotes, rejects."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from greeks_server.wire import Quote, Trade, decode_tick, parse_market_ts

MS = int(datetime(2026, 10, 20, 14, 30, 0, 123000, tzinfo=timezone.utc).timestamp() * 1000)
NOW_MS = MS + 5_000


def test_timestamp_formats() -> None:
    assert parse_market_ts(str(MS)) == MS                              # Massive WS: epoch millis string
    assert parse_market_ts("2026-10-20T14:30:00.123Z") == MS           # REST baseline Instant
    assert parse_market_ts("2026-10-20T14:30:00.123456789Z") == MS     # Alpaca RFC 3339, nanos
    assert parse_market_ts("2026-10-20T10:30:00.123-04:00") == MS
    assert parse_market_ts(MS) == MS


def test_timestamp_rejects() -> None:
    for bad in (None, "", "  ", "yesterday", "2026-10-20T14:30:00", 0, "0", True, 1.5,
                "1792501800", str(MS * 1_000_000)):  # seconds, nanos: not epoch millis
        assert parse_market_ts(bad) is None, bad


def test_option_quote_stream_shape() -> None:
    p = {"type": "quote", "symbol": "NVDA261120C00180000", "bid": 5.1, "ask": 5.3,
         "bidSize": 4, "askSize": 9, "timestamp": str(MS)}
    t = decode_tick(json.dumps(p), NOW_MS)
    assert t is not None and t.symbol == "NVDA261120C00180000"
    assert t.quote == Quote(5.1, 5.3, MS) and t.trade is None


def test_one_sided_quote_kept_with_missing_side() -> None:
    t = decode_tick('{"type":"quote","symbol":"x","bid":NaN,"ask":0.05,"timestamp":"%d"}' % MS, NOW_MS)
    assert t is not None and t.quote == Quote(None, 0.05, MS)


def test_trade_rest_shape() -> None:
    p = {"type": "trade", "symbol": "NVDA", "price": 181.2, "timestamp": "2026-10-20T14:30:00.123Z"}
    t = decode_tick(json.dumps(p), NOW_MS)
    assert t is not None and t.trade == Trade(181.2, MS) and t.quote is None


def test_rejects() -> None:
    assert decode_tick("not json", NOW_MS) is None
    assert decode_tick("[]", NOW_MS) is None
    assert decode_tick('{"type":"quote","symbol":"X","bid":1,"ask":2}', NOW_MS) is None           # no ts
    assert decode_tick('{"type":"trade","symbol":"X","price":0,"timestamp":"%d"}' % MS, NOW_MS) is None
    assert decode_tick('{"type":"bar","symbol":"X","timestamp":"%d"}' % MS, NOW_MS) is None
    assert decode_tick('{"type":"quote","bid":1,"ask":2,"timestamp":"%d"}' % MS, NOW_MS) is None   # no symbol


def test_future_dated_tick_rejected() -> None:
    ahead = '{"type":"quote","symbol":"X","bid":1,"ask":2,"timestamp":"%d"}'
    assert decode_tick(ahead % (NOW_MS + 61_000), NOW_MS) is None
    assert decode_tick(ahead % (NOW_MS + 59_000), NOW_MS) is not None
