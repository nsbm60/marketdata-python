"""The cold path: price a contract the book has never seen, by asking MDS (greeks-server-spec.md §6).

get_greeks answers from the book. A contract not in the book has no held quote or held stock, so the
server asks MDS and prices from that — it never relays a price a client sent. For each contract:

  - get_option_snapshots gives the contract's latest quote; the option price is the quote mid, used
    only if the quote is *current* — two-sided and timestamped at or after the session open + 1
    minute (09:31 ET). The session open comes from the server's own calendar (the most recent
    session open at or before now, the same source compute_greeks uses), not MDS's session_open_ts.
    Otherwise the contract fails no_current_quote, with no stock lookup;
  - get_stock_quotes ("at": "latest") gives each underlying's latest stock quote. The spot is that
    quote's mid (PriceKind.MID, timed at its quote_ts) if two-sided — never a trade. An underlying
    MDS can't quote fails missing_spot for its contracts. Every current contract is priced as of the
    one as_of the stock reply carries, used only for time to expiry — neither the option nor the
    stock quote timestamp is compared against it (ensuring a quote is current is MDS's job).

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

import time

from greeks.domain import FailureReason
from greeks_server.calendar import SessionCache
from greeks_server.compute import Contract, Methodology, _failure, price_current
from greeks_server.inputs import InputsHolder, SessionInputs
from greeks_server.message import OptionGreeksMessage
from greeks_server.price import CURRENT_QUOTE_MARGIN_MS, SelectedPrice, quote_mid
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
    def __init__(self, transport: Transport, inputs: InputsHolder, m: Methodology,
                 sessions: Optional[SessionCache] = None, now_ms: Optional[Callable[[], int]] = None) -> None:
        self._send = transport
        self._inputs = inputs
        self._m = m
        # One source of the session open: the server's own calendar, referenced by now (the same
        # meaning compute_greeks gets from its request as_of) — not MDS's session_open_ts.
        self._sessions = sessions or SessionCache()
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))

    def price(self, osis: Sequence[str]) -> list[ColdResult]:
        """Price each osi from MDS (see the module docstring). One get_option_snapshots call for the
        quotes, then one get_stock_quotes call for the distinct underlyings of the current contracts;
        every current contract is priced as of the stock reply's as_of. The session open — the
        current-quote lower bound — comes from the calendar (most recent session open at or before
        now), the same source compute_greeks uses."""
        contracts = {osi: Contract.parse(osi) for osi in osis}  # raises on a non-standard OSI

        try:
            snaps, _mds_session_open = self._option_snapshots(list(osis))
        except Exception as e:  # noqa: BLE001 — MDS failure: every contract depended on this call
            log.debug("get_option_snapshots failed for %d symbols: %s", len(osis), e)
            results = [ColdResult(osi, None, _failure(contracts[osi], FailureReason.MDS_UNAVAILABLE, 0, self._m, None, None))
                       for osi in osis]
            return self._finish(results)

        # The current-quote lower bound decides which underlyings to fetch stock for; price_current
        # makes the final call once as_of is known (re-checking two-sided + stamped at or before it).
        # No session (None) → nothing is current → every contract is no_current_quote.
        session_open_ms = self._sessions.most_recent_open(self._now_ms())
        threshold = None if session_open_ms is None else session_open_ms + CURRENT_QUOTE_MARGIN_MS
        by_osi: dict[str, ColdResult] = {}
        current: list[tuple[str, Optional[Quote]]] = []
        for osi in osis:
            q, _trade = snaps.get(osi, (None, None))
            mid = quote_mid(q)
            if threshold is not None and mid is not None and mid.ts_ms >= threshold:
                current.append((osi, q))  # candidate: price it once the stock + as_of arrive
            else:
                # No current option quote (one-sided, or before open + 1 min, or no session): no
                # stock lookup. price_current with the epoch as_of yields no_current_quote.
                by_osi[osi] = ColdResult(osi, q, price_current(
                    contracts[osi], q, None, _EPOCH, session_open_ms, _no_inputs(), self._m))

        if current:
            underlyings = sorted({contracts[osi].underlying for osi, _ in current})
            try:
                stock_quotes, as_of_ms = self._stock_quotes(underlyings)
            except Exception as e:  # noqa: BLE001 — MDS failure: only the current contracts depended
                log.debug("get_stock_quotes failed for %d underlyings: %s", len(underlyings), e)
                for osi, q in current:
                    opt_px = quote_mid(q)
                    ts = opt_px.ts_ms if opt_px is not None else 0
                    by_osi[osi] = ColdResult(osi, q, _failure(
                        contracts[osi], FailureReason.MDS_UNAVAILABLE, ts, self._m, opt_px, None))
            else:
                as_of = datetime.fromtimestamp(as_of_ms / 1000.0, tz=timezone.utc)
                inputs = self._inputs.for_date(as_of.astimezone(ET).date())  # one pass, one inputs load
                for osi, q in current:
                    und_q = stock_quotes.get(contracts[osi].underlying)  # None -> missing_spot
                    by_osi[osi] = ColdResult(osi, q, price_current(
                        contracts[osi], q, und_q, as_of, session_open_ms, inputs, self._m))

        return self._finish([by_osi[osi] for osi in osis])

    def stock_prices(self, pairs: Sequence[tuple[str, int]]) -> dict[tuple[str, int], SelectedPrice]:
        """Latest stock mids for the given (underlying, T) pairs, via one get_stock_quotes call. Used
        by the streaming resolver when the option price is already known (a stream tick, a held
        failure) and only the stock is missing. The T is the caller's pricing key; the stock is the
        latest quote mid regardless, so each pair maps to its underlying's latest mid (if any)."""
        unique = sorted(set(pairs))
        underlyings = sorted({u for u, _ in unique})
        quotes, _as_of = self._stock_quotes(underlyings)
        spots = {u: mid for u, q in quotes.items() if (mid := quote_mid(q)) is not None}
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

    def _stock_quotes(self, underlyings: list[str]) -> tuple[dict[str, Quote], int]:
        """One get_stock_quotes ("at": "latest") call for the given underlyings. Returns (per-symbol
        stock quote, reply as_of). The raw quote is returned (not yet mid'd) so price_current applies
        the one spot rule — two-sided mid, stamped at or before as_of, never a trade. An underlying
        MDS can't quote is simply absent (price_current then yields missing_spot for its contracts)."""
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
        out: dict[str, Quote] = {}
        for sym, state in quotes.items():
            if isinstance(sym, str):
                q = _stock_quote_of(state)
                if q is not None:
                    out[sym] = q
        log.debug("get_stock_quotes: %d requested, %d returned (as_of=%d)", len(underlyings), len(out), as_of_ms)
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
