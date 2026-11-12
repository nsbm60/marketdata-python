"""The one pricing rule (compute.price_current) that the cold path and compute_greeks share.

Priced as of the request's as_of: the option price is the opt_quote mid (two-sided, at or before
as_of, at or after the open + 1 min); the spot is the und_quote mid (two-sided, at or before as_of);
never a trade. The cold path and compute_greeks give identical results for the same quotes and time.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from greeks.config import load_config
from greeks_server.coldpath import ColdPath
from greeks_server.handler import make_handler
from greeks_server.inputs import InputsHolder, SessionInputs
from greeks_server.loop import methodology

ET = ZoneInfo("America/New_York")
DAY = date(2026, 11, 13)  # Friday, before the 2026-11-20 expiry; EST
M = methodology(load_config())  # the same methodology make_handler uses, so the cross-check matches
OSI = "NVDA261120C00180000"


def et_ms(h: int, m: int, s: int = 0, day: date = DAY) -> int:
    return int(datetime.combine(day, time(h, m, s), tzinfo=ET).astimezone(timezone.utc).timestamp() * 1000)


SESSION_OPEN = et_ms(9, 30)
AS_OF = et_ms(11, 0)


def _loader(d: date) -> SessionInputs:
    return SessionInputs(session_date=d, sofr={d - timedelta(days=i): 0.043 for i in range(1, 20)}, dividends={})


def _handler() -> Any:
    return make_handler(load_config(), InputsHolder(_loader))


def _req(contract: dict[str, Any], as_of: int = AS_OF) -> dict[str, Any]:
    return {"op": "compute_greeks", "as_of": as_of, "contracts": [contract]}


def _one(contract: dict[str, Any], as_of: int = AS_OF) -> dict[str, Any]:
    return _handler()(_req(contract, as_of))["results"][0]


def _oq(bid: float, ask: float, ts: int) -> dict[str, Any]:
    return {"bid": bid, "ask": ask, "quote_ts": ts}


def test_option_quote_before_open_plus_one_minute_is_no_current_quote() -> None:
    res = _one({"osi": OSI, "opt_quote": _oq(9.0, 9.2, et_ms(9, 30, 40)), "und_quote": _oq(180.9, 181.1, AS_OF)})
    assert res["status"] == "failure" and res["reason"] == "no_current_quote"


def test_one_sided_stock_quote_is_missing_spot() -> None:
    res = _one({"osi": OSI, "opt_quote": _oq(9.0, 9.2, AS_OF), "und_quote": _oq(0.0, 181.1, AS_OF)})  # no bid
    assert res["status"] == "failure" and res["reason"] == "missing_spot"


def test_stock_quote_after_as_of_is_missing_spot() -> None:
    res = _one({"osi": OSI, "opt_quote": _oq(9.0, 9.2, AS_OF), "und_quote": _oq(180.9, 181.1, AS_OF + 60_000)})
    assert res["status"] == "failure" and res["reason"] == "missing_spot"


def test_a_trade_with_no_stock_quote_is_missing_spot() -> None:
    # und_trade is accepted on the wire but ignored: with no und_quote the spot is missing.
    res = _one({"osi": OSI, "opt_quote": _oq(9.0, 9.2, AS_OF), "und_trade": {"last": 181.0, "trade_ts": AS_OF}})
    assert res["status"] == "failure" and res["reason"] == "missing_spot"


def test_a_valid_pair_is_priced_as_of_as_of() -> None:
    res = _one({"osi": OSI, "opt_quote": _oq(9.0, 9.2, AS_OF - 5_000), "und_quote": _oq(180.9, 181.1, AS_OF - 5_000)})
    assert res["status"] == "success"
    assert res["priced_as_of"] == AS_OF                              # one time for the request
    assert res["price_kind"] == "mid" and res["price"] == 9.1
    assert res["spot_kind"] == "mid" and res["spot"] == 181.0


def _cold_transport(opt_quote: dict[str, Any], und_quote: dict[str, Any]):
    """A ColdPath transport that serves the same opt_quote + stock quote + as_of as the handler sees,
    with session_open_ts = the open compute_greeks derives for AS_OF's day."""
    def send(req: dict[str, Any]) -> dict[str, Any]:
        if req["op"] == "get_option_snapshots":
            return {"ok": True, "data": {"snapshots": {OSI: {"quote": {"bid": opt_quote["bid"], "ask": opt_quote["ask"],
                                                                        "at": opt_quote["quote_ts"]}}},
                                         "session_open_ts": SESSION_OPEN}}
        if req["op"] == "get_stock_quotes":
            return {"ok": True, "data": {"quotes": {"NVDA": und_quote}, "missing": [], "as_of": AS_OF}}
        raise AssertionError(f"unexpected op {req['op']}")
    return send


def test_cold_path_and_compute_greeks_agree_for_the_same_quotes_and_time() -> None:
    opt_quote = _oq(9.0, 9.2, AS_OF - 5_000)
    und_quote = _oq(180.9, 181.1, AS_OF - 5_000)
    cold = ColdPath(_cold_transport(opt_quote, und_quote), InputsHolder(_loader), M)

    cold_wire = cold.price([OSI])[0].message.to_wire()
    hot_wire = _one({"osi": OSI, "opt_quote": opt_quote, "und_quote": und_quote})
    assert cold_wire == hot_wire
    assert cold_wire["status"] == "success" and cold_wire["priced_as_of"] == AS_OF
