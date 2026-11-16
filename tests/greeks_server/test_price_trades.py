"""price_trades: the archive's insert-time op. Its results equal greeks.trades.price_trade_rows field
for field, every whole-request rejection is covered, and an un-loadable day fails with the loader's
error rather than returning empty greeks.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pytest

from greeks.config import load_config
from greeks.domain import Dividend
from greeks.trades import price_trade_rows
from greeks_server.handler import (
    OP_TRADES,
    _trade_result,
    make_trades_handler,
    parse_trades_request,
)
from greeks_server.inputs import InputsHolder, SessionInputs

from gs_helpers import NOW_MS, SESSION, sofr_flat

_DIV = {"NVDA": (Dividend("NVDA", 0.25, date(2026, 11, 5)),)}


def _holder() -> InputsHolder:
    def loader(d: date) -> SessionInputs:
        return SessionInputs(session_date=d, sofr=sofr_flat(), dividends=_DIV)
    return InputsHolder(loader)


def _trade(osi: str, price: float, spot: float | None, ts: int = NOW_MS) -> dict[str, Any]:
    return {"osi": osi, "price": price, "spot": spot, "trade_ts": ts}


def test_price_trades_equals_price_trade_rows() -> None:
    cfg = load_config()
    holder = _holder()
    trades = [
        _trade("NVDA261120C00180000", 9.1, 181.0),    # success call
        _trade("NVDA261120P00180000", 8.0, 181.0),    # success put (dividend in window)
        _trade("B1VDA261120C00180000", 9.1, 181.0),   # non-standard OSI -> nonstandard_contract
        _trade("NVDA261120C00180000", 9.1, None),     # null spot -> missing_spot
        _trade("NVDA260101C00180000", 9.1, 181.0),    # expiry before session -> t_below_floor
        _trade("NVDA261120C00100000", 1.0, 181.0),    # below intrinsic
        _trade("NVDA261120C00180000", 1e6, 181.0),    # above no-arb
    ]
    req = {"op": OP_TRADES, "session_date": SESSION.isoformat(), "trades": trades}

    reply = make_trades_handler(cfg, holder)(req)

    # Reference: the same trades straight through the core (greeks values are now-independent, so the
    # two now's don't matter; _trade_result drops the top-level computed_at).
    sd, rows = parse_trades_request(req)
    cols, _ = price_trade_rows(
        rows, holder.for_date(sd), session_date=sd, methodology_version=cfg.methodology_version,
        t_floor_minutes=float(cfg.t_floor_minutes), expiry_time_et=cfg.expiry_time_et,
        now=datetime.now(timezone.utc))
    assert reply["results"] == [_trade_result(c) for c in cols]  # field for field

    assert reply["methodology"] == cfg.methodology_version
    assert reply["basis"] == "trade"
    assert isinstance(reply["computed_at"], int) and reply["computed_at"] > 0
    assert len(reply["results"]) == len(trades)
    assert reply["results"][0]["status"] == "success" and reply["results"][0]["iv"] is not None
    reasons = {r["reason"] for r in reply["results"]}
    assert None in reasons  # a success
    assert {"nonstandard_contract", "missing_spot", "t_below_floor", "below_intrinsic",
            "above_no_arb"} <= reasons


_OK_TRADE = {"osi": "NVDA261120C00180000", "price": 9.1, "spot": 181.0, "trade_ts": NOW_MS}


@pytest.mark.parametrize("payload, msg", [
    ({"trades": []}, "'session_date'"),                                                    # missing
    ({"session_date": 20261020, "trades": []}, "'session_date'"),                          # wrong type
    ({"session_date": "2026-13-40", "trades": []}, "session_date"),                        # not a date
    ({"session_date": "2026-10-20", "trades": "x"}, "'trades' must be a list"),            # not a list
    ({"session_date": "2026-10-20", "trades": ["x"]}, "trades[0] must be an object"),      # not an object
    ({"session_date": "2026-10-20",
      "trades": [{"osi": "X", "price": 1.0, "spot": 1.0}]}, "missing field(s): trade_ts"),  # missing field
    ({"session_date": "2026-10-20",
      "trades": [{**_OK_TRADE, "extra": 1}]}, "unknown field(s): extra"),                  # unknown field
    ({"session_date": "2026-10-20",
      "trades": [{**_OK_TRADE, "osi": 5}]}, "osi must be a string"),                       # osi not a string
    ({"session_date": "2026-10-20",
      "trades": [{**_OK_TRADE, "price": "nan"}]}, "price must be a finite number"),        # price not a number
    ({"session_date": "2026-10-20",
      "trades": [{**_OK_TRADE, "price": float("inf")}]}, "price must be a finite number"),  # price non-finite
    ({"session_date": "2026-10-20",
      "trades": [{**_OK_TRADE, "spot": "x"}]}, "spot must be a finite number"),            # spot not finite/null
    ({"session_date": "2026-10-20",
      "trades": [{**_OK_TRADE, "trade_ts": 0}]}, "trade_ts must be epoch millis"),         # non-positive
    ({"session_date": "2026-10-20",
      "trades": [{**_OK_TRADE, "trade_ts": 1.5}]}, "trade_ts must be epoch millis"),       # not an int
])
def test_price_trades_whole_request_rejections(payload: dict[str, Any], msg: str) -> None:
    with pytest.raises(ValueError) as e:
        parse_trades_request(payload)
    assert msg in str(e.value)


def test_null_spot_is_a_per_row_failure_not_a_request_error() -> None:
    # A null spot is accepted at the edge and becomes MISSING_SPOT in its position.
    _sd, rows = parse_trades_request(
        {"session_date": "2026-10-20", "trades": [{**_OK_TRADE, "spot": None}]})
    assert rows[0].spot is None


def test_price_trades_fails_if_inputs_cannot_load() -> None:
    cfg = load_config()

    def bad_loader(_d: date) -> SessionInputs:
        raise RuntimeError("no SOFR or T-bill for that day")

    req = {"session_date": SESSION.isoformat(), "trades": [_OK_TRADE]}
    with pytest.raises(RuntimeError, match="no SOFR"):
        make_trades_handler(cfg, InputsHolder(bad_loader))(req)
