"""The two control ops (greeks-server-spec.md §4, §6).

compute_greeks — raw-facts, pure. Prices each contract from the facts in the request and returns
the results; it never touches the book or publishes. Used by tools/Tradeability.scala.

  Request:  {"op": "compute_greeks", "as_of": <epoch ms>,
             "contracts": [{"osi",
                            "opt_quote": {"bid","ask","quote_ts"}?,  "opt_trade": {"last","trade_ts"}?,
                            "und_quote": {"bid","ask","quote_ts"}?,  "und_trade": {"last","trade_ts"}?}, …]}
  Response: {"results": [<OptionGreeksMessage wire dict>, …]}  — one per input, input order.

  Pricing follows the same rule the cold path uses (compute.price_current), one time for the whole
  request: every contract is priced as of the request's ``as_of`` — used only for time to expiry
  (and its date for the rate and dividends).
    - Option price: the opt_quote mid, valid only if two-sided and stamped at or after the open + 1
      minute of as_of's session (09:31 ET). Otherwise no_current_quote.
    - Spot: the und_quote mid, valid only if two-sided. Otherwise missing_spot. Never a trade.
    - Neither price is compared against as_of: ensuring a quote is as of the right time is the data
      source's job (as_of is a computer clock, quote times are the exchange's — comparing them
      rejects harmless millisecond skew).
    - Rate and dividends are for as_of's date.
  opt_trade and und_trade are still accepted on the wire (so existing callers don't break) but are
  IGNORED — greeks price from quote mids only.

get_greeks — the live book op. Answers each contract from the book, prices any the book has never
seen through the cold path (which asks MDS), and publishes those.

  Request:  {"op": "get_greeks", "contracts": [<osi>, …]}
  Response: {"results": [<OptionGreeksMessage wire dict>, …]}  — one per contract, input order.

Validation is at the edge: a malformed request is rejected whole with a specific error. Nothing is
defaulted.

A requested symbol that is a string but not a standard OSI (e.g. an adjusted/binary root like
B1VDA…) is NOT a whole-request error: it becomes an ``invalid_symbol`` failure in its input
position and the rest of the request is priced. In get_greeks such a symbol never reaches the book,
the cold path, or publication — only the valid symbols go to ComputeLoop.get_greeks, merged back in
input order. Still rejected whole (unchanged): contracts missing/not a list; an entry of the wrong
type (compute_greeks: not an object; get_greeks: not a string); an osi that isn't a string; unknown
or missing fields; a bad as_of.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Union
from zoneinfo import ZoneInfo

from greeks.config import GreeksConfig
from greeks.domain import FailureReason, RowStatus
from greeks_server.calendar import SessionCache
from greeks_server.compute import Contract, Methodology, price_current
from greeks_server.inputs import InputsHolder
from greeks_server.loop import methodology
from greeks_server.message import OptionGreeksMessage
from greeks_server.wire import Quote, Trade

log = logging.getLogger(__name__)

OP = "compute_greeks"
OP_GET = "get_greeks"
ET = ZoneInfo("America/New_York")
_CONTRACT_FIELDS = ("osi", "opt_quote", "opt_trade", "und_quote", "und_trade")
_QUOTE_FIELDS = ("bid", "ask", "quote_ts")
_TRADE_FIELDS = ("last", "trade_ts")

# One compute_greeks input: the contract and the raw facts for it (each side present or absent).
ParsedItem = tuple[Contract, Optional[Quote], Optional[Trade], Optional[Quote], Optional[Trade]]


@dataclass(frozen=True)
class InvalidSymbol:
    """A requested symbol that is a string but not a standard OSI — it fails on its own."""
    osi: str


def _invalid_message(osi: str, priced_as_of_ms: int, m: Methodology) -> OptionGreeksMessage:
    """The invalid_symbol failure, built directly (not via _failure, which needs a parsed Contract).
    Every price and greek field is empty."""
    return OptionGreeksMessage(
        osi=osi,
        status=RowStatus.FAILURE,
        reason=FailureReason.INVALID_SYMBOL,
        priced_as_of=priced_as_of_ms,
        methodology_version=m.methodology_version,
    )


def _warn_invalid(op: str, invalid: list[str]) -> None:
    """One WARN per request that has invalid symbols: the op, the count, up to 5 examples."""
    if invalid:
        log.warning("%s: %d invalid symbol(s) failed on their own (e.g. %s)",
                    op, len(invalid), ", ".join(invalid[:5]))


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


def parse_request(payload: Mapping[str, Any]) -> tuple[int, list[Union[ParsedItem, InvalidSymbol]]]:
    as_of = payload.get("as_of")
    if isinstance(as_of, bool) or not isinstance(as_of, int) or as_of <= 0:
        raise ValueError(f"'as_of' must be epoch millis (positive int), got {as_of!r}")
    contracts = payload.get("contracts")
    if not isinstance(contracts, list):
        raise ValueError("'contracts' must be a list")
    out: list[Union[ParsedItem, InvalidSymbol]] = []
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
        # Validate the facts first, so a malformed quote/trade still rejects the whole request even
        # on an entry whose osi doesn't parse.
        oq = _quote(f"contracts[{i}].opt_quote", item.get("opt_quote"))
        ot = _trade(f"contracts[{i}].opt_trade", item.get("opt_trade"))
        uq = _quote(f"contracts[{i}].und_quote", item.get("und_quote"))
        ut = _trade(f"contracts[{i}].und_trade", item.get("und_trade"))
        try:
            contract = Contract.parse(osi)
        except ValueError:
            out.append(InvalidSymbol(osi))  # not a standard OSI: fails on its own, in position
            continue
        out.append((contract, oq, ot, uq, ut))
    return as_of, out


def make_handler(cfg: GreeksConfig, inputs: InputsHolder) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """compute_greeks: pure, raw-facts. No book, no publish. Prices every contract as of the
    request's as_of via the shared compute.price_current rule (see the module docstring)."""
    m = methodology(cfg)
    sessions = SessionCache()  # the one source of the session open (same as the cold path uses)

    def handle(payload: Mapping[str, Any]) -> dict[str, Any]:
        as_of_ms, items = parse_request(payload)  # validate everything before any work
        # One time for the whole request: the as_of, its session open, and its day's inputs. The
        # session open is the most recent regular session that opened at or before as_of (None if
        # none can be placed → every contract is no_current_quote); never a naive 09:30 of as_of's day.
        as_of = datetime.fromtimestamp(as_of_ms / 1000.0, tz=timezone.utc)
        session_open_ms = sessions.most_recent_open(as_of_ms)
        day_inputs = inputs.for_date(as_of.astimezone(ET).date())
        results = []
        invalid: list[str] = []
        for it in items:
            if isinstance(it, InvalidSymbol):
                invalid.append(it.osi)
                results.append(_invalid_message(it.osi, priced_as_of_ms=as_of_ms, m=m).to_wire())
                continue
            contract, oq, _ot, uq, _ut = it  # opt_trade/und_trade accepted on the wire but ignored
            results.append(price_current(contract, oq, uq, as_of, session_open_ms, day_inputs, m).to_wire())
        _warn_invalid(OP, invalid)
        return {"results": results}

    return handle


def parse_get_request(payload: Mapping[str, Any]) -> list[Union[str, InvalidSymbol]]:
    contracts = payload.get("contracts")
    if not isinstance(contracts, list):
        raise ValueError("'contracts' must be a list")
    out: list[Union[str, InvalidSymbol]] = []
    for i, osi in enumerate(contracts):
        if not isinstance(osi, str):
            raise ValueError(f"contracts[{i}] must be a string osi, got {osi!r}")
        try:
            Contract.parse(osi)
            out.append(osi)
        except ValueError:
            out.append(InvalidSymbol(osi))  # not a standard OSI: fails on its own, in position
    return out


def make_get_handler(
    get_greeks: Callable[[list[str]], list[OptionGreeksMessage]],
    m: Methodology,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """get_greeks: book first, cold path for the rest. ``get_greeks`` is ComputeLoop.get_greeks.
    A non-OSI symbol fails on its own (invalid_symbol, priced_as_of 0 as the cold path's failures
    use today); only the valid symbols reach ``get_greeks``, merged back in input order."""

    def handle(payload: Mapping[str, Any]) -> dict[str, Any]:
        items = parse_get_request(payload)
        valid = [it for it in items if isinstance(it, str)]
        priced = iter(get_greeks(valid))  # one per valid symbol, in order
        results = []
        invalid: list[str] = []
        for it in items:
            if isinstance(it, InvalidSymbol):
                invalid.append(it.osi)
                results.append(_invalid_message(it.osi, priced_as_of_ms=0, m=m).to_wire())
            else:
                results.append(next(priced).to_wire())
        _warn_invalid(OP_GET, invalid)
        return {"results": results}

    return handle
