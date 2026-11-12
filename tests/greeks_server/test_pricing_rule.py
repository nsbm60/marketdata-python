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
from greeks_server.calendar import SessionCache
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


# ── one session open for both callers (calendar lookup, not a naive 09:30 of as_of's day) ───────

# Scenarios where a naive 09:30-of-as_of's-day differs from the most recent session open.
_FRI_0800 = et_ms(8, 0)                                  # pre-market on a trading day (Fri 2026-11-13)
_SATURDAY = et_ms(12, 0, day=date(2026, 11, 14))         # weekend
_THANKSGIVING = et_ms(12, 0, day=date(2026, 11, 26))     # holiday
_THU_OPEN = et_ms(9, 30, day=date(2026, 11, 12))         # the session before Fri 08:00
_FRI_OPEN = et_ms(9, 30, day=date(2026, 11, 13))         # the session before Saturday
_WED_OPEN = et_ms(9, 30, day=date(2026, 11, 25))         # the session before Thanksgiving


def test_most_recent_session_open_uses_the_prior_session_not_a_naive_0930() -> None:
    sc = SessionCache()
    assert sc.most_recent_open(_FRI_0800) == _THU_OPEN       # 08:00 on a trading day -> previous session
    assert sc.most_recent_open(_SATURDAY) == _FRI_OPEN       # Saturday -> Friday
    assert sc.most_recent_open(_THANKSGIVING) == _WED_OPEN   # holiday -> the last session before it


def _cold_transport(osi: str, opt_quote: dict[str, Any], und_quote: dict[str, Any], as_of: int):
    """A ColdPath transport serving the same opt_quote + stock quote + as_of the handler sees. The
    cold path derives the session open from its own calendar (now_ms), so session_open_ts is unused."""
    def send(req: dict[str, Any]) -> dict[str, Any]:
        if req["op"] == "get_option_snapshots":
            return {"ok": True, "data": {"snapshots": {osi: {"quote": {"bid": opt_quote["bid"], "ask": opt_quote["ask"],
                                                                       "at": opt_quote["quote_ts"]}}},
                                         "session_open_ts": 0}}
        if req["op"] == "get_stock_quotes":
            return {"ok": True, "data": {"quotes": {"NVDA": und_quote}, "missing": [], "as_of": as_of}}
        raise AssertionError(f"unexpected op {req['op']}")
    return send


def _agree(as_of: int, quote_ts: int, osi: str = OSI) -> dict[str, Any]:
    """Price one contract both ways at `as_of` (cold path's clock = as_of), assert identical wire."""
    opt_quote = _oq(9.0, 9.2, quote_ts)
    und_quote = _oq(180.9, 181.1, quote_ts)
    cold = ColdPath(_cold_transport(osi, opt_quote, und_quote, as_of), InputsHolder(_loader), M, now_ms=lambda: as_of)
    cold_wire = cold.price([osi])[0].message.to_wire()
    hot_wire = _one({"osi": osi, "opt_quote": opt_quote, "und_quote": und_quote}, as_of=as_of)
    assert cold_wire == hot_wire
    return cold_wire


def test_cold_path_and_compute_greeks_agree_pre_market_on_a_trading_day() -> None:
    # as_of Fri 08:00: both gate on Thu's open, so a Thu-10:00 quote is current and prices.
    wire = _agree(_FRI_0800, quote_ts=et_ms(10, 0, day=date(2026, 11, 12)))
    assert wire["status"] == "success" and wire["priced_as_of"] == _FRI_0800


def test_cold_path_and_compute_greeks_agree_on_a_weekend() -> None:
    # as_of Saturday: both gate on Friday's open, so a Fri-10:00 quote is current and prices.
    wire = _agree(_SATURDAY, quote_ts=et_ms(10, 0, day=date(2026, 11, 13)))
    assert wire["status"] == "success" and wire["priced_as_of"] == _SATURDAY


def test_cold_path_and_compute_greeks_agree_on_a_holiday() -> None:
    # as_of Thanksgiving: both gate on Wednesday's open, so a Wed-10:00 quote is current and prices.
    # A December-expiry contract, since Thanksgiving is after the 2026-11-20 expiry.
    wire = _agree(_THANKSGIVING, quote_ts=et_ms(10, 0, day=date(2026, 11, 25)), osi="NVDA261218C00180000")
    assert wire["status"] == "success" and wire["priced_as_of"] == _THANKSGIVING
