"""Price selection against the shared golden cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from greeks_server.price import PriceKind, select_price, select_spot
from greeks_server.wire import Quote, Trade

GOLDEN = json.loads((Path(__file__).parent / "price_golden.json").read_text(encoding="utf-8"))
OPEN = GOLDEN["session_open_ms"]
PRICE_TIME = GOLDEN["price_time_ms"]


def _quote(d: Optional[dict[str, Any]]) -> Optional[Quote]:
    return None if d is None else Quote(d["bid"], d["ask"], d["ts"])


def _trade(d: Optional[dict[str, Any]]) -> Optional[Trade]:
    return None if d is None else Trade(d["price"], d["ts"])


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_golden(case: dict[str, Any]) -> None:
    got = select_price(_quote(case["quote"]), _trade(case["trade"]), OPEN, PRICE_TIME)
    exp = case["expect"]
    if exp is None:
        assert got is None
    else:
        assert got is not None
        assert got.kind is PriceKind(exp["kind"])
        assert got.value == pytest.approx(exp["value"], abs=1e-12)
        assert got.ts_ms == exp["ts"]


def test_select_spot_prefers_quote_then_newest_valid_trade() -> None:
    und_trade = Trade(100.0, 4500)   # after-hours-ish but within window here
    und_close = Trade(99.0, 2000)    # the regular-session close
    # No quote: the newer valid trade wins.
    s = select_spot(None, (und_trade, und_close), OPEN, PRICE_TIME)
    assert s is not None and s.kind is PriceKind.LAST and s.value == 100.0 and s.ts_ms == 4500
    # A valid quote beats both trades.
    s2 = select_spot(Quote(101.0, 101.2, 3000), (und_trade, und_close), OPEN, PRICE_TIME)
    assert s2 is not None and s2.kind is PriceKind.MID


def test_select_spot_falls_back_to_close_when_later_trade_is_out_of_window() -> None:
    # The underlying's latest trade is after price-time (an after-hours print); only the close is
    # valid, so the spot is the regular-session close — the off-hours fix.
    after_hours = Trade(188.70, 6000)   # > price_time
    close = Trade(186.64, 4800)
    s = select_spot(None, (after_hours, close), OPEN, PRICE_TIME)
    assert s is not None and s.value == 186.64 and s.ts_ms == 4800
