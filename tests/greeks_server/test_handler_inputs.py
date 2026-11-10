"""compute_greeks (raw-facts, pure): request validation, the as_of upper bound, inputs by date."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from greeks.config import load_config
from greeks_server.handler import make_handler, parse_request
from greeks_server.inputs import InputsHolder, SessionInputs, validate_sofr

from gs_helpers import NOW_MS, SESSION, sofr_flat


def _item(**over: Any) -> dict[str, Any]:
    # Raw facts: a two-sided option quote (mid 9.1) and an underlying last trade. An absent side is
    # omitted. select_price picks the mid for the option (fixing T at its quote_ts); select_spot
    # picks the last for the spot.
    d: dict[str, Any] = {
        "osi": "NVDA261120C00180000",
        "opt_quote": {"bid": 9.0, "ask": 9.2, "quote_ts": NOW_MS},
        "und_trade": {"last": 181.0, "trade_ts": NOW_MS},
    }
    d.update(over)
    return d


def _req(contracts: list[dict[str, Any]], as_of: int = NOW_MS) -> dict[str, Any]:
    return {"op": "compute_greeks", "as_of": as_of, "contracts": contracts}


def _handler(loads: list[date]) -> Any:
    def loader(d: date) -> SessionInputs:
        loads.append(d)
        return SessionInputs(session_date=d, sofr=sofr_flat(), dividends={})
    return make_handler(load_config(), InputsHolder(loader))


def test_results_in_input_order_and_inputs_cached_by_date() -> None:
    loads: list[date] = []
    h = _handler(loads)
    deep_itm = _item(osi="NVDA261120C00100000", opt_quote={"bid": 0.9, "ask": 1.1, "quote_ts": NOW_MS})
    out = h(_req([_item(), deep_itm]))
    res = out["results"]
    assert [r["osi"] for r in res] == ["NVDA261120C00180000", "NVDA261120C00100000"]
    assert res[0]["status"] == "success" and res[1]["reason"] == "below_intrinsic"
    assert res[0]["spot_kind"] == "last" and res[0]["price_kind"] == "mid"  # the server ran the §4 rule
    # Both price as of the same date; inputs are loaded once for it.
    assert loads == [SESSION]


def test_priced_as_of_is_the_option_price_timestamp_not_as_of() -> None:
    # T is the option price's own timestamp, at or before as_of.
    item = _item(
        opt_quote={"bid": 9.0, "ask": 9.2, "quote_ts": NOW_MS - 5_000},
        und_trade={"last": 181.0, "trade_ts": NOW_MS - 5_000},
    )
    res = _handler([])(_req([item], as_of=NOW_MS))["results"]
    assert res[0]["status"] == "success"
    assert res[0]["priced_as_of"] == NOW_MS - 5_000


def test_no_trade_fallback() -> None:
    # Only a trade, no two-sided quote: no_current_quote (the trade fallback is gone).
    item = {"osi": "NVDA261120C00180000",
            "opt_trade": {"last": 9.05, "trade_ts": NOW_MS},
            "und_trade": {"last": 181.0, "trade_ts": NOW_MS}}
    res = _handler([])(_req([item], as_of=NOW_MS))["results"]
    assert res[0]["status"] == "failure" and res[0]["reason"] == "no_current_quote"


def test_a_quote_after_as_of_is_not_current() -> None:
    # The option quote is the only price; stamped after as_of it is rejected -> no_current_quote.
    item = _item(opt_quote={"bid": 9.0, "ask": 9.2, "quote_ts": NOW_MS + 60_000})
    res = _handler([])(_req([item], as_of=NOW_MS))["results"]
    assert res[0]["status"] == "failure" and res[0]["reason"] == "no_current_quote"


def test_empty_list_is_fine() -> None:
    assert _handler([])(_req([])) == {"results": []}


@pytest.mark.parametrize(
    "bad, msg",
    [
        (_req("x"), "'contracts' must be a list"),  # type: ignore[arg-type]
        ({"op": "compute_greeks", "contracts": [_item()]}, "'as_of' must be epoch millis"),
        (_req([_item(osi="NVDA1261120C00180000")]), "not a standard OSI"),
        (_req([_item(opt_trade={"last": 0, "trade_ts": NOW_MS})]), "opt_trade.last must be positive"),
        (_req([_item(und_quote={"bid": float("inf"), "ask": 1.0, "quote_ts": NOW_MS})]), "und_quote.bid must be a finite number"),
        (_req([_item(opt_quote={"bid": 9.0, "quote_ts": NOW_MS})]), "missing field(s): ask"),
        (_req([_item(opt_quote={"bid": 9.0, "ask": 9.2, "quote_ts": 1.5})]), "opt_quote.quote_ts must be epoch millis"),
        (_req([_item(extra=1)]), "unknown field(s): extra"),
        (_req([_item(und_close={"last": 1.0, "trade_ts": NOW_MS})]), "unknown field(s): und_close"),  # und_close dropped
        (_req([_item(opt_quote={"bid": 9.0, "ask": 9.2, "quote_ts": NOW_MS, "x": 1})]), "unknown field(s): x"),
    ],
)
def test_rejects_whole_request(bad: dict[str, Any], msg: str) -> None:
    with pytest.raises(ValueError) as e:
        parse_request(bad)
    assert msg in str(e.value)


def test_validate_sofr() -> None:
    validate_sofr(sofr_flat(), SESSION)
    with pytest.raises(ValueError, match="more than 10 days"):
        validate_sofr({SESSION - timedelta(days=11): 0.04}, SESSION)
    with pytest.raises(ValueError, match="no SOFR fix"):
        validate_sofr({SESSION + timedelta(days=1): 0.04}, SESSION)


def test_rate_memo_per_expiry() -> None:
    inp = SessionInputs(session_date=SESSION, sofr=sofr_flat(0.04), dividends={})
    r1 = inp.rate_for(date(2026, 11, 20))
    assert inp.rate_for(date(2026, 11, 20)) is r1
    assert r1 == pytest.approx(0.04 * 365 / 360, rel=1e-3)  # ACT/360 simple -> ACT/365 continuous


def test_inputs_holder_loads_once_per_date() -> None:
    loads: list[date] = []

    def loader(d: date) -> SessionInputs:
        loads.append(d)
        return SessionInputs(session_date=d, sofr=sofr_flat(), dividends={})

    h = InputsHolder(loader)
    a = h.for_date(SESSION)
    assert h.for_date(SESSION) is a                       # cached
    b = h.for_date(SESSION + timedelta(days=1))
    assert b is not a
    assert loads == [SESSION, SESSION + timedelta(days=1)]
