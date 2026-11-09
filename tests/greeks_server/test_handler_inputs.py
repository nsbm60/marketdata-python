"""compute_greeks request validation and results; SessionInputs and SOFR checks."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from greeks.config import load_config
from greeks_server.handler import make_handler, parse_request
from greeks_server.inputs import InputsHolder, SessionInputs, validate_sofr

from gs_helpers import CLOSE_MS, NOW_MS, SESSION, SESSION_OBJ, sofr_flat


def _item(**over: Any) -> dict[str, Any]:
    # Raw facts: a two-sided option quote (mid 9.1) and an underlying last trade. An absent side
    # (here opt_trade and und_quote) is simply omitted. select_price picks mid for the option and
    # last for the spot, matching the pre-selected (9.1 mid, 181 last) the old request sent.
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
    return make_handler(load_config(), InputsHolder(loader), lambda t: SESSION_OBJ)


def test_results_in_input_order() -> None:
    loads: list[date] = []
    h = _handler(loads)
    deep_itm = _item(osi="NVDA261120C00100000", opt_quote={"bid": 0.9, "ask": 1.1, "quote_ts": NOW_MS})
    out = h(_req([_item(), deep_itm]))
    res = out["results"]
    assert [r["osi"] for r in res] == ["NVDA261120C00180000", "NVDA261120C00100000"]
    assert res[0]["status"] == "success" and res[1]["reason"] == "below_intrinsic"
    assert res[0]["spot_kind"] == "last" and res[0]["price_kind"] == "mid"  # server ran the §4 rule
    h(_req([_item()]))
    # as_of is in session but nothing loaded the session: each request loads fresh, uncached,
    # so the open can't inherit a pre-08:45 load.
    assert loads == [SESSION, SESSION]


def test_off_session_prices_at_close() -> None:
    # as_of after the close: the server prices at the session's close, so computed_at is the close.
    loads: list[date] = []
    out = _handler(loads)(_req([_item()], as_of=CLOSE_MS + 3_600_000))
    res = out["results"]
    assert res[0]["computed_at"] == CLOSE_MS
    assert loads == [SESSION]  # off-session request loads fresh, uncached


def test_off_session_uses_und_close_over_after_hours_trade() -> None:
    # The off-hours bug: the underlying's latest trade is a 20:00 after-hours print (188.70), after
    # the close — it must be rejected. und_close (186.64, the 16:00 close) is used for the spot.
    item = _item(
        osi="NVDA261120C00180000",
        opt_quote={"bid": 9.0, "ask": 9.2, "quote_ts": CLOSE_MS - 60_000},
        und_trade={"last": 188.70, "trade_ts": CLOSE_MS + 60_000},   # after close → rejected
        und_close={"last": 186.64, "trade_ts": CLOSE_MS - 60_000},   # the regular-session close
    )
    out = _handler([])(_req([item], as_of=CLOSE_MS + 3_600_000))
    res = out["results"]
    assert res[0]["status"] == "success"
    assert res[0]["spot"] == 186.64 and res[0]["spot_kind"] == "last"  # the close, not 188.70


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


def test_holder_session_vs_request() -> None:
    loads: list[date] = []

    def loader(d: date) -> SessionInputs:
        loads.append(d)
        return SessionInputs(session_date=d, sofr=sofr_flat(), dividends={})

    h = InputsHolder(loader)
    h.for_request(SESSION, in_session=False)
    assert h.current() is None                     # off-session request does not cache
    s1 = h.session_inputs(SESSION)                 # the open loads
    assert h.for_request(SESSION, in_session=True) is s1
    assert h.session_inputs(SESSION) is s1
    assert loads == [SESSION, SESSION]
