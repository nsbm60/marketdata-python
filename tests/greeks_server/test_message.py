"""OptionGreeksMessage: golden round trip, omission of absent fields, invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from greeks.domain import FailureReason, RowStatus
from greeks_server.message import OptionGreeksMessage
from greeks_server.price import PriceKind

GOLDEN = json.loads((Path(__file__).parent / "option_greeks_golden.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("wire", GOLDEN["messages"], ids=lambda w: f"{w['osi']}-{w.get('reason', 'ok')}")
def test_golden_round_trip(wire: dict[str, Any]) -> None:
    msg = OptionGreeksMessage.from_wire(wire)
    assert msg.to_wire() == wire
    assert json.loads(json.dumps(msg.to_wire())) == wire
    assert all(v is not None for v in msg.to_wire().values())


def test_golden_decodes_to_expected_values() -> None:
    ok = OptionGreeksMessage.from_wire(GOLDEN["messages"][0])
    assert ok.status is RowStatus.SUCCESS and ok.price_kind is PriceKind.MID and ok.gamma == 0.0123
    nsp = OptionGreeksMessage.from_wire(GOLDEN["messages"][1])
    assert nsp.reason is FailureReason.NO_SESSION_PRICE and nsp.price is None and nsp.spot == 196.1


def _ok(**over: Any) -> dict[str, Any]:
    d = dict(GOLDEN["messages"][0])
    d.update(over)
    return d


def test_success_requires_all_greeks() -> None:
    d = _ok()
    del d["gamma"]
    with pytest.raises(ValueError):
        OptionGreeksMessage.from_wire(d)


def test_failure_cannot_carry_greeks() -> None:
    d = dict(GOLDEN["messages"][1])
    d["iv"] = 0.3
    with pytest.raises(ValueError):
        OptionGreeksMessage.from_wire(d)


def test_price_fields_go_together() -> None:
    d = _ok()
    del d["price_ts"]
    with pytest.raises(ValueError):
        OptionGreeksMessage.from_wire(d)


def test_strict_decode() -> None:
    with pytest.raises(ValueError):
        OptionGreeksMessage.from_wire(_ok(extra=1))
    with pytest.raises(ValueError):
        OptionGreeksMessage.from_wire(_ok(price_ts="1792501799870"))
    with pytest.raises(ValueError):
        OptionGreeksMessage.from_wire(_ok(price_kind="bid"))
