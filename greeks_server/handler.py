"""The two control ops (greeks-server-spec.md §4, §6).

compute_greeks — raw-facts, pure. Prices each contract from the facts in the request and returns
the results; it never touches the book or publishes. Used by tools/Tradeability.scala.

  Request:  {"op": "compute_greeks", "as_of": <epoch ms>,
             "contracts": [{"osi",
                            "opt_quote": {"bid","ask","quote_ts"}?,  "opt_trade": {"last","trade_ts"}?,
                            "und_quote": {"bid","ask","quote_ts"}?,  "und_trade": {"last","trade_ts"}?}, …]}
  Response: {"results": [<OptionGreeksMessage wire dict>, …]}  — one per input, input order.

  as_of is an upper bound: for each contract T is the option price's own timestamp at or before
  as_of (mid first, else last), the spot is the newest fact at or before T, and the rate and
  dividends are for T's date. No session, no clock.

get_greeks — the live book op. Answers each contract from the book, prices any the book has never
seen through the cold path (which asks MDS), and publishes those.

  Request:  {"op": "get_greeks", "contracts": [<osi>, …]}
  Response: {"results": [<OptionGreeksMessage wire dict>, …]}  — one per contract, input order.

Validation is at the edge: a malformed request is rejected whole with a specific error. Nothing is
defaulted.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional
from zoneinfo import ZoneInfo

from greeks.config import GreeksConfig
from greeks_server.compute import Contract, compute_or_fail
from greeks_server.inputs import InputsHolder
from greeks_server.loop import methodology
from greeks_server.message import OptionGreeksMessage
from greeks_server.price import select_price, select_spot
from greeks_server.wire import Quote, Trade

OP = "compute_greeks"
OP_GET = "get_greeks"
ET = ZoneInfo("America/New_York")
_CONTRACT_FIELDS = ("osi", "opt_quote", "opt_trade", "und_quote", "und_trade")
_QUOTE_FIELDS = ("bid", "ask", "quote_ts")
_TRADE_FIELDS = ("last", "trade_ts")

# One compute_greeks input: the contract and the raw facts for it (each side present or absent).
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


def make_handler(cfg: GreeksConfig, inputs: InputsHolder) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """compute_greeks: pure, raw-facts. No book, no publish, no clock."""
    m = methodology(cfg)

    def handle(payload: Mapping[str, Any]) -> dict[str, Any]:
        as_of_ms, items = parse_request(payload)  # validate everything before any work
        results = []
        for contract, oq, ot, uq, ut in items:
            opt_px = select_price(oq, ot, as_of_ms)
            t_ms = opt_px.ts_ms if opt_px is not None else as_of_ms
            spot_px = select_spot(uq, (ut,), t_ms)
            as_of = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
            day_inputs = inputs.for_date(as_of.astimezone(ET).date())
            results.append(compute_or_fail(contract, opt_px, spot_px, day_inputs, as_of, m).to_wire())
        return {"results": results}

    return handle


def parse_get_request(payload: Mapping[str, Any]) -> list[str]:
    contracts = payload.get("contracts")
    if not isinstance(contracts, list):
        raise ValueError("'contracts' must be a list")
    osis: list[str] = []
    for i, osi in enumerate(contracts):
        if not isinstance(osi, str):
            raise ValueError(f"contracts[{i}] must be a string osi, got {osi!r}")
        try:
            Contract.parse(osi)  # reject a non-standard OSI at the edge
        except ValueError:
            raise ValueError(f"contracts[{i}] is not a standard OSI symbol: {osi!r}") from None
        osis.append(osi)
    return osis


def make_get_handler(
    get_greeks: Callable[[list[str]], list[OptionGreeksMessage]],
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """get_greeks: book first, cold path for the rest. ``get_greeks`` is ComputeLoop.get_greeks."""

    def handle(payload: Mapping[str, Any]) -> dict[str, Any]:
        osis = parse_get_request(payload)
        return {"results": [m.to_wire() for m in get_greeks(osis)]}

    return handle
