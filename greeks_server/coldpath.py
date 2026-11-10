"""The cold path: price a contract the book has never seen, by asking MDS (greeks-server-spec.md §6).

get_greeks answers from the book. A contract not in the book has no held quote or held stock, so the
server asks MDS and prices from that — it never relays a price a client sent. For each contract:

  - get_option_snapshots gives the contract's latest quote and the session open; the option price is
    the quote mid, used only if the quote is current (at or after the session open) — else a
    no_current_quote failure, with no stock lookup. T is the quote's own timestamp;
  - get_equity_prices_at gives the stock's newest clean print at or before T, for current quotes.

An MDS failure raises — the caller turns it into a failure result naming the reason, never a default.
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
from greeks_server.price import PriceKind, SelectedPrice, quote_mid
from greeks_server.wire import Quote

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
_EPOCH = datetime.fromtimestamp(0, tz=timezone.utc)

# A control transport: send a request dict, get the parsed reply dict. Raises on timeout/transport.
Transport = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ColdResult:
    """One priced contract and the quote it was priced from, so the book can hold the quote for the
    next time-event pass."""
    osi: str
    quote: Optional[Quote]
    message: OptionGreeksMessage


class ColdPath:
    def __init__(self, transport: Transport, inputs: InputsHolder, m: Methodology) -> None:
        self._send = transport
        self._inputs = inputs
        self._m = m

    def price(self, osis: Sequence[str]) -> list[ColdResult]:
        """Price each osi from MDS. `get_option_snapshots` gives the latest quote and the session
        open; the option price is the quote mid, and only if the quote is *current* (at or after the
        session open) — a stale or one-sided quote is a no_current_quote failure, with no stock
        lookup. Current quotes' stock comes from `get_equity_prices_at` at each quote's T."""
        contracts = {osi: Contract.parse(osi) for osi in osis}  # raises on a non-standard OSI
        snaps, session_open_ms = self._option_snapshots(list(osis))

        # Current quotes only; gather their (underlying, T) to ask the stock for.
        plans: list[tuple[str, Optional[Quote], Optional[SelectedPrice], Optional[int]]] = []
        wanted: set[tuple[str, int]] = set()
        for osi in osis:
            q, _trade = snaps.get(osi, (None, None))
            opt_px = quote_mid(q)
            if opt_px is not None and opt_px.ts_ms >= session_open_ms:
                wanted.add((contracts[osi].underlying, opt_px.ts_ms))
                plans.append((osi, q, opt_px, opt_px.ts_ms))
            else:
                plans.append((osi, q, None, None))  # no current quote -> no_current_quote, no stock
        stock = self._equity_prices_at(sorted(wanted))

        results: list[ColdResult] = []
        for osi, q, opt_px, t_ms in plans:
            contract = contracts[osi]
            if t_ms is None:
                # No current two-sided quote: compute() returns no_current_quote; no spot is needed.
                msg = compute_or_fail(contract, None, None, _no_inputs(), _EPOCH, self._m)
                results.append(ColdResult(osi, q, msg))
                continue
            as_of = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
            spot_px = stock.get((contract.underlying, t_ms))
            inputs = self._inputs.for_date(as_of.astimezone(ET).date())
            msg = compute_or_fail(contract, opt_px, spot_px, inputs, as_of, self._m)
            results.append(ColdResult(osi, q, msg))
        return results

    def stock_prices(self, pairs: Sequence[tuple[str, int]]) -> dict[tuple[str, int], SelectedPrice]:
        """MDS get_equity_prices_at for the given (underlying, T) pairs, in one batched call. Used
        when the option price is already known (a stream tick, a held failure) and only the stock is
        missing — so there is no need to re-fetch the option snapshot."""
        return self._equity_prices_at(sorted(set(pairs)))

    # ── MDS calls ──────────────────────────────────────────────────────────────────────────────

    def _option_snapshots(self, osis: list[str]) -> tuple[dict[str, tuple[Optional[Quote], None]], int]:
        """Returns (per-osi (quote, None), session_open_ts). The option trade is not read — greeks
        price from the quote mid only. session_open_ts gates whether a quote is current (§4); it is
        0 only if an older MDS omitted it (no gate — the pre-change behaviour)."""
        data = self._data({"op": "get_option_snapshots", "symbols": osis}, "get_option_snapshots")
        snaps = data.get("snapshots")
        if not isinstance(snaps, dict):
            raise RuntimeError("get_option_snapshots reply has no snapshots object")
        so = data.get("session_open_ts")
        session_open_ms = so if isinstance(so, int) and not isinstance(so, bool) else 0
        out: dict[str, tuple[Optional[Quote], None]] = {}
        for osi, state in snaps.items():
            if isinstance(state, dict):
                out[osi] = (_quote_of(state.get("quote")), None)
        return out, session_open_ms

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


def _quote_of(v: Any) -> Optional[Quote]:
    if not isinstance(v, dict):
        return None
    at = v.get("at")
    if not isinstance(at, int):
        return None
    bid, ask = v.get("bid"), v.get("ask")
    return Quote(float(bid) if isinstance(bid, (int, float)) else None,
                 float(ask) if isinstance(ask, (int, float)) else None, at)


# A pricing where no current quote arrived: compute() returns no_current_quote before it reads
# inputs, so the value is never used — but the signature needs one.
def _no_inputs() -> SessionInputs:
    return SessionInputs(session_date=_EPOCH.date(), sofr={}, dividends={})
