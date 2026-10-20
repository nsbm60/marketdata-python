"""compute_greeks request validation and results; SessionInputs and SOFR checks."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from greeks.config import load_config
from greeks_server.handler import make_handler, parse_request
from greeks_server.inputs import InputsHolder, SessionInputs, validate_sofr

from gs_helpers import NOW, NOW_MS, SESSION, sofr_flat


def _item(**over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "osi": "NVDA261120C00180000", "price": 9.1, "price_kind": "mid", "price_ts": NOW_MS,
        "spot": 181.0, "spot_kind": "last", "spot_ts": NOW_MS,
    }
    d.update(over)
    return d


def _handler(loads: list[date]) -> Any:
    def loader(d: date) -> SessionInputs:
        loads.append(d)
        return SessionInputs(session_date=d, sofr=sofr_flat(), dividends={})
    return make_handler(load_config(), InputsHolder(loader), lambda t: object(), now=lambda: NOW)  # type: ignore[arg-type,return-value]


def test_results_in_input_order() -> None:
    loads: list[date] = []
    h = _handler(loads)
    out = h({"op": "compute_greeks", "contracts": [_item(), _item(osi="NVDA261120C00100000", price=1.0)]})
    res = out["results"]
    assert [r["osi"] for r in res] == ["NVDA261120C00180000", "NVDA261120C00100000"]
    assert res[0]["status"] == "success" and res[1]["reason"] == "below_intrinsic"
    assert res[0]["spot_kind"] == "last"
    h({"op": "compute_greeks", "contracts": [_item()]})
    # NOW is in session but nothing loaded the session: each request loads fresh, uncached,
    # so the open can't inherit a pre-08:45 load.
    assert loads == [SESSION, SESSION]


def test_empty_list_is_fine() -> None:
    assert _handler([])({"op": "compute_greeks", "contracts": []}) == {"results": []}


@pytest.mark.parametrize(
    "bad, msg",
    [
        ({"contracts": "x"}, "'contracts' must be a list"),
        ({"contracts": [_item(osi="NVDA1261120C00180000")]}, "not a standard OSI"),
        ({"contracts": [_item(price=0)]}, "price must be a positive number"),
        ({"contracts": [_item(spot=float("inf"))]}, "spot must be a positive number"),
        ({"contracts": [_item(price_kind="bid")]}, "price_kind must be 'mid' or 'last'"),
        ({"contracts": [_item(spot_ts=1.5)]}, "spot_ts must be epoch millis"),
        ({"contracts": [{k: v for k, v in _item().items() if k != "spot_kind"}]}, "missing field(s): spot_kind"),
        ({"contracts": [_item(extra=1)]}, "unknown field(s): extra"),
    ],
)
def test_rejects_whole_request(bad: dict[str, Any], msg: str) -> None:
    with pytest.raises(ValueError) as e:
        parse_request({"op": "compute_greeks", **bad})
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
