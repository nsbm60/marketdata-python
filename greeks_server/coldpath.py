"""The cold path: price a contract the book has never seen, by asking MDS (greeks-server-spec.md §6).

get_greeks answers from the book. A contract not in the book has no held quote or held stock, so the
server asks MDS and prices from that — it never relays a price a client sent. For each contract:

  - get_option_snapshots gives the contract's latest quote and the session open; the option price is
    the quote mid, used only if the quote is *current* — two-sided and timestamped at or after the
    session open + 1 minute (09:31 ET). Otherwise the contract fails no_current_quote, with no stock
    lookup;
  - get_stock_quotes ("at": "latest") gives each underlying's latest stock quote. The spot is that
    quote's mid (PriceKind.MID, timed at its quote_ts) — never a trade. An underlying MDS can't quote
    fails missing_spot for its contracts. Every current contract is priced as of the one as_of the
    stock reply carries.

An MDS call that raises or times out fails ONLY the contracts that depended on it: if
get_option_snapshots fails, every contract gets mds_unavailable; if get_stock_quotes fails, only the
current-quote contracts do, and the no_current_quote ones are still answered. The rest of the request
is always answered — a failed contract gets a failure result naming the reason, never a default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence
from zoneinfo import ZoneInfo

from greeks.domain import FailureReason
from greeks_server.compute import Contract, Methodology, _failure, compute_or_fail
from greeks_server.inputs import InputsHolder, SessionInputs
from greeks_server.message import OptionGreeksMessage
from greeks_server.price import PriceKind, SelectedPrice, quote_mid
from greeks_server.wire import Quote

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
_EPOCH = datetime.fromtimestamp(0, tz=timezone.utc)

# A current option quote must be timestamped at or after the session open + this margin (09:31 ET).
_CURRENT_QUOTE_MARGIN_MS = 60_000

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
        """Price each osi from MDS (see the module docstring). One get_option_snapshots call for the
        quotes + session open, then one get_stock_quotes call for the distinct underlyings of the
        current contracts; every current contract is priced as of the stock reply's as_of."""
        contracts = {osi: Contract.parse(osi) for osi in osis}  # raises on a non-standard OSI

        try:
            snaps, session_open_ms = self._option_snapshots(list(osis))
        except Exception as e:  # noqa: BLE001 — MDS failure: every contract depended on this call
            log.debug("get_option_snapshots failed for %d symbols: %s", len(osis), e)
            results = [ColdResult(osi, None, _failure(contracts[osi], FailureReason.MDS_UNAVAILABLE, 0, self._m, None, None))
                       for osi in osis]
            return self._finish(results)

        threshold = session_open_ms + _CURRENT_QUOTE_MARGIN_MS
        by_osi: dict[str, ColdResult] = {}
        current: list[tuple[str, Optional[Quote], SelectedPrice]] = []
        for osi in osis:
            q, _trade = snaps.get(osi, (None, None))
            opt_px = quote_mid(q)
            if opt_px is not None and opt_px.ts_ms >= threshold:
                current.append((osi, q, opt_px))
            else:
                # No current two-sided quote: no_current_quote, and no stock lookup.
                msg = compute_or_fail(contracts[osi], None, None, _no_inputs(), _EPOCH, self._m)
                by_osi[osi] = ColdResult(osi, q, msg)

        if current:
            underlyings = sorted({contracts[osi].underlying for osi, _, _ in current})
            try:
                spots, as_of_ms = self._stock_quotes(underlyings)
            except Exception as e:  # noqa: BLE001 — MDS failure: only the current contracts depended
                log.debug("get_stock_quotes failed for %d underlyings: %s", len(underlyings), e)
                for osi, q, opt_px in current:
                    by_osi[osi] = ColdResult(osi, q, _failure(
                        contracts[osi], FailureReason.MDS_UNAVAILABLE, opt_px.ts_ms, self._m, opt_px, None))
            else:
                as_of = datetime.fromtimestamp(as_of_ms / 1000.0, tz=timezone.utc)
                inputs = self._inputs.for_date(as_of.astimezone(ET).date())  # one pass, one inputs load
                for osi, q, opt_px in current:
                    spot_px = spots.get(contracts[osi].underlying)  # None -> compute() -> missing_spot
                    msg = compute_or_fail(contracts[osi], opt_px, spot_px, inputs, as_of, self._m)
                    by_osi[osi] = ColdResult(osi, q, msg)

        return self._finish([by_osi[osi] for osi in osis])

    def stock_prices(self, pairs: Sequence[tuple[str, int]]) -> dict[tuple[str, int], SelectedPrice]:
        """Latest stock mids for the given (underlying, T) pairs, via one get_stock_quotes call. Used
        by the streaming resolver when the option price is already known (a stream tick, a held
        failure) and only the stock is missing. The T is the caller's pricing key; the stock is the
        latest quote mid regardless, so each pair maps to its underlying's latest mid (if any)."""
        unique = sorted(set(pairs))
        underlyings = sorted({u for u, _ in unique})
        spots, _as_of = self._stock_quotes(underlyings)
        return {(u, t): spots[u] for (u, t) in unique if u in spots}

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
        log.debug("get_option_snapshots: %d requested, %d returned (session_open_ts=%d)",
                  len(osis), len(out), session_open_ms)
        return out, session_open_ms

    def _stock_quotes(self, underlyings: list[str]) -> tuple[dict[str, SelectedPrice], int]:
        """One get_stock_quotes ("at": "latest") call for the given underlyings. Returns (per-symbol
        stock mid, reply as_of). The spot is the quote mid, PriceKind.MID, timed at its quote_ts; an
        underlying with no current two-sided quote is simply absent (its contracts -> missing_spot)."""
        if not underlyings:
            return {}, 0
        data = self._data({"op": "get_stock_quotes", "symbols": underlyings, "at": "latest"}, "get_stock_quotes")
        quotes = data.get("quotes")
        if not isinstance(quotes, dict):
            raise RuntimeError("get_stock_quotes reply has no quotes object")
        as_of = data.get("as_of")
        if not isinstance(as_of, int) or isinstance(as_of, bool) or as_of <= 0:
            # No usable as_of is a malformed reply — never price as of 1970. Raise so the contracts
            # that depend on this call fail mds_unavailable, like any other bad MDS reply.
            raise RuntimeError(f"get_stock_quotes reply has no valid as_of: {as_of!r}")
        as_of_ms = as_of
        out: dict[str, SelectedPrice] = {}
        for sym, state in quotes.items():
            if isinstance(sym, str):
                px = quote_mid(_stock_quote_of(state))
                if px is not None:
                    out[sym] = px
        log.debug("get_stock_quotes: %d requested, %d priced (as_of=%d)", len(underlyings), len(out), as_of_ms)
        return out, as_of_ms

    def _data(self, req: dict[str, Any], op: str) -> dict[str, Any]:
        reply = self._send(req)
        if not isinstance(reply, dict) or not reply.get("ok"):
            raise RuntimeError(f"MDS {op} failed: {reply.get('error') if isinstance(reply, dict) else reply!r}")
        data = reply.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"MDS {op} reply has no data object")
        return data

    def _finish(self, results: list[ColdResult]) -> list[ColdResult]:
        """One WARN per request that has any failures, with counts by reason (the per-call lines are
        DEBUG)."""
        counts: dict[str, int] = {}
        for r in results:
            reason = r.message.reason
            if reason is not None:
                counts[reason.value] = counts.get(reason.value, 0) + 1
        if counts:
            log.warning("cold path: %d/%d contracts failed (%s)", sum(counts.values()), len(results),
                        ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        return results


def _quote_of(v: Any) -> Optional[Quote]:
    if not isinstance(v, dict):
        return None
    at = v.get("at")
    if not isinstance(at, int):
        return None
    bid, ask = v.get("bid"), v.get("ask")
    return Quote(float(bid) if isinstance(bid, (int, float)) else None,
                 float(ask) if isinstance(ask, (int, float)) else None, at)


def _stock_quote_of(v: Any) -> Optional[Quote]:
    """Parse a get_stock_quotes entry {"bid", "ask", "quote_ts"} into a Quote. quote_mid then takes
    the two-sided mid (or None)."""
    if not isinstance(v, dict):
        return None
    ts = v.get("quote_ts")
    if not isinstance(ts, int) or isinstance(ts, bool):
        return None
    bid, ask = v.get("bid"), v.get("ask")
    return Quote(float(bid) if isinstance(bid, (int, float)) and not isinstance(bid, bool) else None,
                 float(ask) if isinstance(ask, (int, float)) and not isinstance(ask, bool) else None, ts)


# A pricing where no current quote arrived: compute() returns no_current_quote before it reads
# inputs, so the value is never used — but the signature needs one.
def _no_inputs() -> SessionInputs:
    return SessionInputs(session_date=_EPOCH.date(), sofr={}, dividends={})
