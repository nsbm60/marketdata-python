"""compute_greeks — the on-demand batch op (greeks-server-spec.md §4, §6).

Request:  {"op": "compute_greeks", "as_of": <epoch ms>,
           "contracts": [{"osi",
                          "opt_quote": {"bid","ask","quote_ts"}?,  "opt_trade": {"last","trade_ts"}?,
                          "und_quote": {"bid","ask","quote_ts"}?,  "und_trade": {"last","trade_ts"}?}, …]}
Response: {"results": [<OptionGreeksMessage wire dict>, …]}  — one per input, input order.

The caller sends raw facts, not a chosen price: the latest option quote/trade and the underlying's
latest quote/trade, each side omitted when absent (never 0), plus the as-of time. The server owns the
session and the price rule: it picks the session containing as_of (or the last before it when as_of
is outside every session), applies select_price (§4) relative to that session's open for the option
and the spot, prices at as_of in session or at that session's close otherwise, and loads SOFR and
dividends for that session's date. The handler never reads the clock.

Validation is at the edge: a malformed request is rejected whole with a specific error (the Router
turns the exception into {ok: false, error}). Nothing is defaulted.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

from greeks.config import GreeksConfig
from greeks_server.calendar import Session
from greeks_server.compute import Contract, compute_or_fail
from greeks_server.inputs import InputsHolder
from greeks_server.loop import methodology
from greeks_server.price import select_price
from greeks_server.wire import Quote, Trade

OP = "compute_greeks"
_CONTRACT_FIELDS = ("osi", "opt_quote", "opt_trade", "und_quote", "und_trade")
_QUOTE_FIELDS = ("bid", "ask", "quote_ts")
_TRADE_FIELDS = ("last", "trade_ts")

# One request input: the contract and the raw facts for it (each side present or absent).
ParsedItem = tuple[Contract, Optional[Quote], Optional[Trade], Optional[Quote], Optional[Trade]]


def _finite(where: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{where} must be a finite number, got {v!r}")
    return float(v)


def _positive(where: str, v: Any) -> float:
    f = _finite(where, v)
    if not f > 0:
        raise ValueError(f"{where} must be positive, got {v!r}")
    return f


def _ts(where: str, v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise ValueError(f"{where} must be epoch millis (positive int), got {v!r}")
    return v


def _known(where: str, item: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    missing = [k for k in fields if k not in item]
    if missing:
        raise ValueError(f"{where} missing field(s): {', '.join(missing)}")
    extra = sorted(set(item) - set(fields))
    if extra:
        raise ValueError(f"{where} unknown field(s): {', '.join(extra)}")


def _quote(where: str, v: Any) -> Optional[Quote]:
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError(f"{where} must be an object")
    _known(where, v, _QUOTE_FIELDS)
    # bid/ask may be <= 0 (a one-sided market is a legal quote); select_price decides its worth.
    return Quote(_finite(f"{where}.bid", v["bid"]), _finite(f"{where}.ask", v["ask"]), _ts(f"{where}.quote_ts", v["quote_ts"]))


def _trade(where: str, v: Any) -> Optional[Trade]:
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError(f"{where} must be an object")
    _known(where, v, _TRADE_FIELDS)
    return Trade(_positive(f"{where}.last", v["last"]), _ts(f"{where}.trade_ts", v["trade_ts"]))


def parse_request(payload: Mapping[str, Any]) -> tuple[int, list[ParsedItem]]:
    as_of = payload.get("as_of")
    if isinstance(as_of, bool) or not isinstance(as_of, int) or as_of <= 0:
        raise ValueError(f"'as_of' must be epoch millis (positive int), got {as_of!r}")
    contracts = payload.get("contracts")
    if not isinstance(contracts, list):
        raise ValueError("'contracts' must be a list")
    out: list[ParsedItem] = []
    for i, item in enumerate(contracts):
        if not isinstance(item, dict):
            raise ValueError(f"contracts[{i}] must be an object")
        extra = sorted(set(item) - set(_CONTRACT_FIELDS))
        if extra:
            raise ValueError(f"contracts[{i}] unknown field(s): {', '.join(extra)}")
        if "osi" not in item:
            raise ValueError(f"contracts[{i}] missing field(s): osi")
        osi = item["osi"]
        if not isinstance(osi, str):
            raise ValueError(f"contracts[{i}].osi must be a string")
        try:
            contract = Contract.parse(osi)
        except ValueError:
            raise ValueError(f"contracts[{i}].osi is not a standard OSI symbol: {osi!r}") from None
        out.append((
            contract,
            _quote(f"contracts[{i}].opt_quote", item.get("opt_quote")),
            _trade(f"contracts[{i}].opt_trade", item.get("opt_trade")),
            _quote(f"contracts[{i}].und_quote", item.get("und_quote")),
            _trade(f"contracts[{i}].und_trade", item.get("und_trade")),
        ))
    return as_of, out


def make_handler(
    cfg: GreeksConfig,
    inputs: InputsHolder,
    session_pick: Callable[[datetime], Optional[Session]],
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """`session_pick` is SessionCache.session_on_or_before: the session for an as-of time, or the
    last before it. The handler reads no clock — as_of comes in the request."""
    m = methodology(cfg)

    def handle(payload: Mapping[str, Any]) -> dict[str, Any]:
        as_of_ms, items = parse_request(payload)  # validate everything before any work
        as_of = datetime.fromtimestamp(as_of_ms / 1000.0, tz=timezone.utc)
        session = session_pick(as_of)
        if session is None:
            raise ValueError(f"no trading session on or before as_of {as_of.isoformat()}")
        in_session = session.contains(as_of)
        price_time = as_of if in_session else session.close_utc
        open_ms = session.open_ms
        session_inputs = inputs.for_request(session.date, in_session=in_session)
        results = []
        for contract, oq, ot, uq, ut in items:
            opt_px = select_price(oq, ot, open_ms)
            spot_px = select_price(uq, ut, open_ms)
            results.append(compute_or_fail(contract, opt_px, spot_px, session_inputs, price_time, m).to_wire())
        return {"results": results}

    return handle
