"""A non-OSI symbol fails only itself; the rest of the request is priced (both ops)."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pytest

from greeks.config import load_config
from greeks.domain import FailureReason, RowStatus
from greeks_server.handler import make_get_handler, make_handler, parse_get_request, parse_request
from greeks_server.inputs import InputsHolder, SessionInputs
from greeks_server.message import OptionGreeksMessage

from gs_helpers import M, NOW_MS, SESSION, sofr_flat

V1 = "NVDA261120C00180000"   # prices to success
V2 = "NVDA261120C00100000"   # deep ITM -> below_intrinsic (still a priced result)
BAD = "B1VDA261120C00180000" # NYSE American binary root — not a standard OSI
BAD2 = "B1SLA261120C00050000"


def _item(osi: str) -> dict[str, Any]:
    return {"osi": osi,
            "opt_quote": {"bid": 9.0, "ask": 9.2, "quote_ts": NOW_MS},
            "und_trade": {"last": 181.0, "trade_ts": NOW_MS}}


def _compute_handler() -> Any:
    def loader(d: date) -> SessionInputs:
        return SessionInputs(session_date=d, sofr=sofr_flat(), dividends={})
    return make_handler(load_config(), InputsHolder(loader))


def _fake_get(calls: list[list[str]]):
    def fake(osis: list[str]) -> list[OptionGreeksMessage]:
        calls.append(list(osis))
        return [OptionGreeksMessage(osi=o, status=RowStatus.FAILURE, reason=FailureReason.NO_CURRENT_QUOTE,
                                    priced_as_of=123, methodology_version="v2") for o in osis]
    return fake


def test_compute_greeks_invalid_symbol_fails_only_itself() -> None:
    h = _compute_handler()
    with_bad = h({"op": "compute_greeks", "as_of": NOW_MS, "contracts": [_item(V1), _item(BAD), _item(V2)]})["results"]
    without = h({"op": "compute_greeks", "as_of": NOW_MS, "contracts": [_item(V1), _item(V2)]})["results"]

    assert [r["osi"] for r in with_bad] == [V1, BAD, V2]                 # input order
    assert with_bad[1]["status"] == "failure" and with_bad[1]["reason"] == "invalid_symbol"
    assert with_bad[1]["priced_as_of"] == NOW_MS                         # compute_greeks: the request's as_of
    assert with_bad[1]["methodology_version"] == with_bad[0]["methodology_version"]
    assert "delta" not in with_bad[1] and "price" not in with_bad[1]     # every price/greek field empty
    assert with_bad[0] == without[0] and with_bad[2] == without[1]       # the others unchanged


def test_get_greeks_invalid_symbol_fails_only_itself() -> None:
    calls: list[list[str]] = []
    h = make_get_handler(_fake_get(calls), M)
    res = h({"op": "get_greeks", "contracts": [V1, BAD, V2]})["results"]

    assert calls == [[V1, V2]]                                           # only the valid symbols reach get_greeks
    assert [r["osi"] for r in res] == [V1, BAD, V2]                      # merged back in input order
    assert res[1]["status"] == "failure" and res[1]["reason"] == "invalid_symbol"
    assert res[1]["priced_as_of"] == 0                                  # cold-path failures use today (0)
    assert res[1]["methodology_version"] == "v2"                        # the methodology passed in
    assert res[0]["priced_as_of"] == 123 and res[2]["priced_as_of"] == 123  # these came from get_greeks


def test_warn_logged_once_with_count_and_examples(caplog: pytest.LogCaptureFixture) -> None:
    h = make_get_handler(_fake_get([]), M)
    with caplog.at_level(logging.WARNING, logger="greeks_server.handler"):
        h({"op": "get_greeks", "contracts": [BAD, BAD2, V1]})
    warns = [r for r in caplog.records if r.name == "greeks_server.handler" and r.levelno == logging.WARNING]
    assert len(warns) == 1
    msg = warns[0].getMessage()
    assert "get_greeks" in msg and "2 invalid" in msg and BAD in msg


def test_no_warn_when_all_valid(caplog: pytest.LogCaptureFixture) -> None:
    h = make_get_handler(_fake_get([]), M)
    with caplog.at_level(logging.WARNING, logger="greeks_server.handler"):
        h({"op": "get_greeks", "contracts": [V1]})
    assert [r for r in caplog.records if r.name == "greeks_server.handler" and r.levelno == logging.WARNING] == []


def test_non_list_contracts_rejects_whole_request_both_ops() -> None:
    with pytest.raises(ValueError, match="'contracts' must be a list"):
        parse_request({"op": "compute_greeks", "as_of": NOW_MS, "contracts": "x"})
    with pytest.raises(ValueError, match="'contracts' must be a list"):
        parse_get_request({"op": "get_greeks", "contracts": 5})
