"""Price selection against the shared golden cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from greeks_server.price import PriceKind, select_price
from greeks_server.wire import Quote, Trade

GOLDEN = json.loads((Path(__file__).parent / "price_golden.json").read_text(encoding="utf-8"))


def _quote(d: Optional[dict[str, Any]]) -> Optional[Quote]:
    return None if d is None else Quote(d["bid"], d["ask"], d["ts"])


def _trade(d: Optional[dict[str, Any]]) -> Optional[Trade]:
    return None if d is None else Trade(d["price"], d["ts"])


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_golden(case: dict[str, Any]) -> None:
    got = select_price(_quote(case["quote"]), _trade(case["trade"]), GOLDEN["session_open_ms"])
    exp = case["expect"]
    if exp is None:
        assert got is None
    else:
        assert got is not None
        assert got.kind is PriceKind(exp["kind"])
        assert got.value == pytest.approx(exp["value"], abs=1e-12)
        assert got.ts_ms == exp["ts"]
