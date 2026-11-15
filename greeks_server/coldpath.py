"""The cold path: price a contract the book has never seen, by asking MDS (greeks-server-spec.md §6).

get_greeks answers from the book. A contract not in the book has no held quote or held stock, so the
server asks MDS and prices from that — it never relays a price a client sent. The cold path makes ONE
call: get_quotes, carrying the requested OSIs plus their distinct underlyings, as of the caller's
``at`` ("latest" or an epoch-ms T). The server makes no session decision of its own — it prices
whatever it is asked for, as of whatever time the caller gave, using the reply's own session marker.

From the one reply:

  - an option is *current* if it is in ``quotes`` (MDS only returns two-sided quotes there) and its
    ``quote_ts`` is at or after ``session_open_ts`` + 1 minute, using the ``session_open_ts`` the
    reply carries. Otherwise the contract fails no_current_quote;
  - the spot is the underlying's quote mid from the same reply. An underlying that came back in
    ``missing`` fails its contracts with missing_spot;
  - every contract is priced (price_current) as of the reply's ``as_of`` — used only for time to
    expiry (and its date for the rate and dividends). Neither the option nor the stock timestamp is
    compared against it (ensuring a quote is current is MDS's job).

If the get_quotes call raises or times out, every contract fails mds_unavailable. The rest is always
answered — a failed contract gets a failure result naming the reason, never a default.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence, Union
from zoneinfo import ZoneInfo

from greeks.domain import FailureReason
from greeks_server.compute import Contract, Methodology, _failure, price_current
from greeks_server.inputs import InputsHolder
from greeks_server.message import OptionGreeksMessage
from greeks_server.price import SelectedPrice, quote_mid
from greeks_server.wire import Quote

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# A control transport: send a request dict, get the parsed reply dict. Raises on timeout/transport.
Transport = Callable[[dict[str, Any]], dict[str, Any]]

# The caller's `at`, already validated at the edge: the string "latest" or an epoch-ms int.
At = Union[str, int]


class MdsRequestError(RuntimeError):
    """A failure of an MDS control request: a transport error, a not-ok reply (e.g. ``unsupported
    op`` during a deploy), or a malformed reply. It is distinct from a programming error so callers
    can treat an MDS restart or outage as recoverable (fail the affected contracts mds_unavailable)
    rather than let it take the greeks server down."""


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

    def price(self, osis: Sequence[str], at: At) -> tuple[list[ColdResult], Optional[int]]:
        """Price each osi from MDS as of ``at`` (see the module docstring), via one get_quotes call
        carrying the OSIs plus their distinct underlyings. Returns (results, priced_as_of) — the
        priced_as_of is the reply's ``as_of`` (None when the call failed or there was nothing to
        price). The current-quote lower bound is the reply's own session_open_ts; the server makes no
        session decision of its own."""
        contracts = {osi: Contract.parse(osi) for osi in osis}  # raises on a non-standard OSI
        if not contracts:
            return [], None

        osi_list = list(dict.fromkeys(osis))                               # requested OSIs, in order
        underlyings = sorted({c.underlying for c in contracts.values()})   # their distinct underlyings
        symbols = osi_list + [u for u in underlyings if u not in contracts]
        t0 = time.perf_counter()

        try:
            quotes, session_open_ms, as_of_ms = self._get_quotes(symbols, at)
        except Exception as e:  # noqa: BLE001 — MDS failure: every contract depended on this one call
            log.debug("cold path get_quotes failed for %d symbols (at=%s): %s", len(symbols), at, e)
            return [ColdResult(osi, None, _failure(contracts[osi], FailureReason.MDS_UNAVAILABLE, 0, self._m, None, None))
                    for osi in osis], None

        as_of = datetime.fromtimestamp(as_of_ms / 1000.0, tz=timezone.utc)
        inputs = self._inputs.for_date(as_of.astimezone(ET).date())  # one pass, one inputs load
        results: list[ColdResult] = []
        for osi in osis:
            c = contracts[osi]
            opt_q = quotes.get(osi)            # two-sided or absent; current_option_mid gates on session_open
            und_q = quotes.get(c.underlying)   # the spot: absent (in `missing`) -> missing_spot
            results.append(ColdResult(osi, opt_q, price_current(c, opt_q, und_q, as_of, session_open_ms, inputs, self._m)))

        self._log(results, at, session_open_ms, as_of_ms, t0)
        return results, as_of_ms

    def stock_prices(self, pairs: Sequence[tuple[str, int]]) -> dict[tuple[str, int], SelectedPrice]:
        """Latest stock mids for the given (underlying, T) pairs, via one get_quotes ("at": "latest")
        call. Used by the streaming resolver when the option price is already known (a stream tick, a
        held failure) and only the stock is missing. The T is the caller's pricing key; the stock is
        the latest quote mid regardless, so each pair maps to its underlying's latest mid (if any)."""
        unique = sorted(set(pairs))
        underlyings = sorted({u for u, _ in unique})
        if not underlyings:
            return {}
        quotes, _session_open, _as_of = self._get_quotes(underlyings, "latest")
        spots = {u: mid for u, q in quotes.items() if (mid := quote_mid(q)) is not None}
        return {(u, t): spots[u] for (u, t) in unique if u in spots}

    # ── MDS call ─────────────────────────────────────────────────────────────────────────────────

    def _get_quotes(self, symbols: list[str], at: At) -> tuple[dict[str, Quote], Optional[int], int]:
        """One get_quotes call. Returns (per-symbol two-sided quote, session_open_ts, as_of). The
        ``at`` is passed through unchanged. session_open_ts is None when the reply omits it (no
        session can be placed -> every option is no_current_quote); as_of must be present and valid
        (a reply without it is malformed — never price as of 1970)."""
        data = self._data({"op": "get_quotes", "symbols": symbols, "at": at}, "get_quotes")
        quotes_raw = data.get("quotes")
        if not isinstance(quotes_raw, dict):
            raise MdsRequestError("get_quotes reply has no quotes object")
        as_of = data.get("as_of")
        if not isinstance(as_of, int) or isinstance(as_of, bool) or as_of <= 0:
            raise MdsRequestError(f"get_quotes reply has no valid as_of: {as_of!r}")
        so = data.get("session_open_ts")
        session_open_ms = so if (isinstance(so, int) and not isinstance(so, bool) and so > 0) else None
        quotes: dict[str, Quote] = {}
        for sym, state in quotes_raw.items():
            if isinstance(sym, str):
                q = _quote_of(state)
                if q is not None:
                    quotes[sym] = q
        return quotes, session_open_ms, as_of

    def _data(self, req: dict[str, Any], op: str) -> dict[str, Any]:
        try:
            reply = self._send(req)
        except MdsRequestError:
            raise
        except Exception as e:  # a transport/timeout failure of the request itself (an MDS outage)
            raise MdsRequestError(f"MDS {op} request failed: {e}") from e
        if not isinstance(reply, dict) or not reply.get("ok"):
            raise MdsRequestError(f"MDS {op} failed: {reply.get('error') if isinstance(reply, dict) else reply!r}")
        data = reply.get("data")
        if not isinstance(data, dict):
            raise MdsRequestError(f"MDS {op} reply has no data object")
        return data

    def _log(self, results: list[ColdResult], at: At, session_open_ms: Optional[int], as_of_ms: int, t0: float) -> None:
        """One DEBUG line per request: counts, the `at`, and the wall time. Per-contract failures
        (no_current_quote, missing_spot) are expected and counted here, never logged line by line."""
        ms = (time.perf_counter() - t0) * 1000.0
        counts: dict[str, int] = {}
        for r in results:
            if r.message.reason is not None:
                counts[r.message.reason.value] = counts.get(r.message.reason.value, 0) + 1
        priced = sum(1 for r in results if r.message.reason is None)
        fails = (" failures=" + ",".join(f"{k}={v}" for k, v in sorted(counts.items()))) if counts else ""
        log.debug("cold path get_quotes: requested=%d priced=%d at=%s session_open_ts=%s as_of=%d %.0fms%s",
                  len(results), priced, at, session_open_ms, as_of_ms, ms, fails)


def _quote_of(v: Any) -> Optional[Quote]:
    """Parse a get_quotes entry {"bid","ask","bid_size","ask_size","quote_ts"} into a Quote. MDS only
    puts two-sided quotes in `quotes`; quote_mid decides the mid."""
    if not isinstance(v, dict):
        return None
    ts = v.get("quote_ts")
    if not isinstance(ts, int) or isinstance(ts, bool):
        return None
    bid, ask = v.get("bid"), v.get("ask")
    return Quote(float(bid) if isinstance(bid, (int, float)) and not isinstance(bid, bool) else None,
                 float(ask) if isinstance(ask, (int, float)) and not isinstance(ask, bool) else None, ts)
