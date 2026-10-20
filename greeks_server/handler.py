"""compute_greeks — the on-demand batch op (greeks-server-spec.md §6).

Request:  {"op": "compute_greeks",
           "contracts": [{"osi", "price", "price_kind", "price_ts", "spot", "spot_kind", "spot_ts"}, …]}
Response: {"results": [<OptionGreeksMessage wire dict>, …]}  — one per input, input order.

The caller (CalcServer's CandidateLoader, at cutover) has already chosen each price with the §4
rule; the server prices what it is given. Validation is at the edge: a malformed request is
rejected whole with a specific error (the Router turns the exception into {ok: false, error}).
Nothing is defaulted.

Uses the session's SOFR and dividends; outside a session it loads them fresh for today's date
(ET) and does not cache them. Stateless otherwise.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

from greeks.config import GreeksConfig
from greeks_server.calendar import ET, Session
from greeks_server.compute import Contract, compute_or_fail
from greeks_server.inputs import InputsHolder
from greeks_server.loop import methodology
from greeks_server.price import PriceKind, SelectedPrice

OP = "compute_greeks"
_FIELDS = ("osi", "price", "price_kind", "price_ts", "spot", "spot_kind", "spot_ts")


def _number(i: int, item: Mapping[str, Any], key: str) -> float:
    v = item[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not v > 0:
        raise ValueError(f"contracts[{i}].{key} must be a positive number, got {v!r}")
    return float(v)


def _ts(i: int, item: Mapping[str, Any], key: str) -> int:
    v = item[key]
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise ValueError(f"contracts[{i}].{key} must be epoch millis (positive int), got {v!r}")
    return v


def _kind(i: int, item: Mapping[str, Any], key: str) -> PriceKind:
    v = item[key]
    try:
        return PriceKind(v)
    except ValueError:
        raise ValueError(f"contracts[{i}].{key} must be 'mid' or 'last', got {v!r}") from None


def parse_request(payload: Mapping[str, Any]) -> list[tuple[Contract, SelectedPrice, SelectedPrice]]:
    contracts = payload.get("contracts")
    if not isinstance(contracts, list):
        raise ValueError("'contracts' must be a list")
    out: list[tuple[Contract, SelectedPrice, SelectedPrice]] = []
    for i, item in enumerate(contracts):
        if not isinstance(item, dict):
            raise ValueError(f"contracts[{i}] must be an object")
        missing = [k for k in _FIELDS if k not in item]
        if missing:
            raise ValueError(f"contracts[{i}] missing field(s): {', '.join(missing)}")
        extra = sorted(set(item) - set(_FIELDS))
        if extra:
            raise ValueError(f"contracts[{i}] unknown field(s): {', '.join(extra)}")
        osi = item["osi"]
        if not isinstance(osi, str):
            raise ValueError(f"contracts[{i}].osi must be a string")
        try:
            contract = Contract.parse(osi)
        except ValueError:
            raise ValueError(f"contracts[{i}].osi is not a standard OSI symbol: {osi!r}") from None
        option_px = SelectedPrice(_number(i, item, "price"), _kind(i, item, "price_kind"), _ts(i, item, "price_ts"))
        spot_px = SelectedPrice(_number(i, item, "spot"), _kind(i, item, "spot_kind"), _ts(i, item, "spot_ts"))
        out.append((contract, option_px, spot_px))
    return out


def make_handler(
    cfg: GreeksConfig,
    inputs: InputsHolder,
    session_fn: Callable[[datetime], Optional[Session]],
    *,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    m = methodology(cfg)

    def handle(payload: Mapping[str, Any]) -> dict[str, Any]:
        items = parse_request(payload)  # validate everything before any work
        t = now()
        session_inputs = inputs.for_request(t.astimezone(ET).date(), in_session=session_fn(t) is not None)
        return {
            "results": [
                compute_or_fail(c, opx, spx, session_inputs, t, m).to_wire() for c, opx, spx in items
            ]
        }

    return handle
