"""The cold path: price a contract the book has never seen, by asking MDS (greeks-server-spec.md §6).

get_greeks answers from the book. A contract not in the book has no held quote or stock history, so
the server asks MDS for the two facts it needs and prices from them — it never relays a price a
client sent. For each contract:

  - get_option_snapshots gives the contract's latest quote and trade; the option price (mid first,
    else last) fixes T, its own timestamp;
  - get_equity_prices_at gives the stock's newest clean print at or before T.

Then it prices as of T, exactly as a live tick would. Batched: one get_option_snapshots for the
whole request, one get_equity_prices_at for every (underlying, T) in it. An MDS failure raises — the
caller turns it into a failure result naming the reason, never a default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence
from zoneinfo import ZoneInfo

from greeks_server.compute import Contract, Methodology, compute_or_fail
from greeks_server.inputs import InputsHolder, SessionInputs
from greeks_server.message import OptionGreeksMessage
from greeks_server.price import PriceKind, SelectedPrice, select_price
from greeks_server.wire import Quote, Trade

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
_LATEST = 2**62  # "at or before" bound that admits every held timestamp (pick the newest usable)
_EPOCH = datetime.fromtimestamp(0, tz=timezone.utc)

# A control transport: send a request dict, get the parsed reply dict. Raises on timeout/transport.
Transport = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ColdResult:
    """One priced contract and the raw facts it was priced from, so the book can hold them for the
    next time-event pass."""
    osi: str
    quote: Optional[Quote]
    trade: Optional[Trade]
    message: OptionGreeksMessage


class ColdPath:
    def __init__(self, transport: Transport, inputs: InputsHolder, m: Methodology) -> None:
        self._send = transport
        self._inputs = inputs
        self._m = m

    def price(self, osis: Sequence[str]) -> list[ColdResult]:
        """Price each osi from MDS: the option's latest quote/trade (`get_option_snapshots`) fixes T,
        then the stock at T (`get_equity_prices_at`, one batched call). Used for contracts the live
        bus has not given the server — so it never pairs with held live stock, which is "now", not T."""
        contracts = {osi: Contract.parse(osi) for osi in osis}  # raises on a non-standard OSI
        snaps = self._option_snapshots(list(osis))

        # T per contract: the option price's own timestamp (mid first). Gather the (underlying, T)
        # pairs to ask the stock for in one call.
        plans: list[tuple[str, Optional[Quote], Optional[Trade], Optional[SelectedPrice], Optional[int]]] = []
        wanted: set[tuple[str, int]] = set()
        for osi in osis:
            q, t = snaps.get(osi, (None, None))
            opt_px = select_price(q, t, _LATEST)
            t_ms = opt_px.ts_ms if opt_px is not None else _latest_ts(q, t)
            plans.append((osi, q, t, opt_px, t_ms))
            if t_ms is not None:
                wanted.add((contracts[osi].underlying, t_ms))
        stock = self._equity_prices_at(sorted(wanted))

        results: list[ColdResult] = []
        for osi, q, t, opt_px, t_ms in plans:
            contract = contracts[osi]
            if t_ms is None:
                # No quote and no trade from MDS: nothing to price as of. Report the absence.
                log.warning("cold path: no option quote or trade for %s", osi)
                msg = compute_or_fail(contract, None, None, _no_inputs(), _EPOCH, self._m)
                results.append(ColdResult(osi, q, t, msg))
                continue
            as_of = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
            spot_px = stock.get((contract.underlying, t_ms))
            inputs = self._inputs.for_date(as_of.astimezone(ET).date())
            msg = compute_or_fail(contract, opt_px, spot_px, inputs, as_of, self._m)
            results.append(ColdResult(osi, q, t, msg))
        return results

    def stock_prices(self, pairs: Sequence[tuple[str, int]]) -> dict[tuple[str, int], SelectedPrice]:
        """MDS get_equity_prices_at for the given (underlying, T) pairs, in one batched call. Used
        when the option price is already known (a stream tick, a held failure) and only the stock is
        missing — so there is no need to re-fetch the option snapshot."""
        return self._equity_prices_at(sorted(set(pairs)))

    # ── MDS calls ──────────────────────────────────────────────────────────────────────────────

    def _option_snapshots(self, osis: list[str]) -> dict[str, tuple[Optional[Quote], Optional[Trade]]]:
        data = self._data({"op": "get_option_snapshots", "symbols": osis}, "get_option_snapshots")
        snaps = data.get("snapshots")
        if not isinstance(snaps, dict):
            raise RuntimeError("get_option_snapshots reply has no snapshots object")
        out: dict[str, tuple[Optional[Quote], Optional[Trade]]] = {}
        for osi, state in snaps.items():
            if not isinstance(state, dict):
                continue
            out[osi] = (_quote_of(state.get("quote")), _trade_of(state.get("trade")))
        return out

    def _equity_prices_at(self, pairs: list[tuple[str, int]]) -> dict[tuple[str, int], SelectedPrice]:
        if not pairs:
            return {}
        req = {"op": "get_equity_prices_at", "symbols_at": [{"symbol": s, "ts": ts} for s, ts in pairs]}
        data = self._data(req, "get_equity_prices_at")
        prices = data.get("prices")
        if not isinstance(prices, list):
            raise RuntimeError("get_equity_prices_at reply has no prices list")
        out: dict[tuple[str, int], SelectedPrice] = {}
        for p in prices:
            if not isinstance(p, dict):
                continue
            sym, as_of, last, ts = p.get("symbol"), p.get("as_of"), p.get("last"), p.get("trade_ts")
            if isinstance(sym, str) and isinstance(as_of, int) and isinstance(last, (int, float)) and isinstance(ts, int):
                out[(sym, as_of)] = SelectedPrice(float(last), PriceKind.LAST, ts)
        return out

    def _data(self, req: dict[str, Any], op: str) -> dict[str, Any]:
        reply = self._send(req)
        if not isinstance(reply, dict) or not reply.get("ok"):
            raise RuntimeError(f"MDS {op} failed: {reply.get('error') if isinstance(reply, dict) else reply!r}")
        data = reply.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"MDS {op} reply has no data object")
        return data


def _latest_ts(q: Optional[Quote], t: Optional[Trade]) -> Optional[int]:
    stamps = [x.ts_ms for x in (q, t) if x is not None]
    return max(stamps) if stamps else None


def _quote_of(v: Any) -> Optional[Quote]:
    if not isinstance(v, dict):
        return None
    at = v.get("at")
    if not isinstance(at, int):
        return None
    bid, ask = v.get("bid"), v.get("ask")
    return Quote(float(bid) if isinstance(bid, (int, float)) else None,
                 float(ask) if isinstance(ask, (int, float)) else None, at)


def _trade_of(v: Any) -> Optional[Trade]:
    if not isinstance(v, dict):
        return None
    price, at = v.get("price"), v.get("at")
    if isinstance(price, (int, float)) and price > 0 and isinstance(at, int):
        return Trade(float(price), at)
    return None


# A pricing where no fact arrived: compute() returns missing_spot before it reads inputs, so the
# value is never used — but the signature needs one.
def _no_inputs() -> SessionInputs:
    return SessionInputs(session_date=_EPOCH.date(), sofr={}, dividends={})
