"""ComputeLoop.step(): the three triggers, book ordering, expiry drop, get_greeks + cold path.

Driven directly through step() — no sockets, no threads. There is no session and no clock: every T
comes from the data, so the tests assert priced_as_of against the timestamps they feed in.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pytest

from greeks.config import load_config
from greeks.domain import FailureReason, RowStatus
from greeks.topic_builder import TopicBuilder
from greeks_server.coldpath import ColdPath
from greeks_server.compute import Methodology
from greeks_server.inputs import InputsHolder, SessionInputs
from greeks_server.loop import ComputeLoop
from greeks_server.message import OptionGreeksMessage
from greeks_server.price import PriceKind, SelectedPrice
from greeks_server.wire import Quote

ET = ZoneInfo("America/New_York")
DAY = date(2026, 11, 13)  # a Friday, well before the 2026-11-20 expiry; EST (UTC-5)
NVDA_C = "NVDA261120C00180000"
NVDA_P = "NVDA261120P00170000"
OPT_C = "md.option.quote.NVDA.2026-11-20.C.180_00"
OPT_P = "md.option.quote.NVDA.2026-11-20.P.170_00"


def et_ms(h: int, m: int, s: int = 0, day: date = DAY) -> int:
    return int(datetime.combine(day, time(h, m, s), tzinfo=ET).astimezone(timezone.utc).timestamp() * 1000)


def _sofr(d: date) -> dict[date, float]:
    return {d - timedelta(days=i): 0.043 for i in range(1, 20)}


class Harness:
    def __init__(self, cold: Any = None, loader_fails: bool = False) -> None:
        self.cfg = replace(load_config(), server=replace(load_config().server, topic_prefix="shadow."))
        self.published: list[tuple[str, dict[str, Any]]] = []
        self.loads: list[date] = []
        self.fatal: list[BaseException] = []
        self.now_dt = datetime(2026, 11, 13, 23, 59, tzinfo=timezone.utc)  # late enough: no tick is "future"

        def loader(d: date) -> SessionInputs:
            self.loads.append(d)
            if loader_fails:
                raise ValueError("no SOFR")
            return SessionInputs(session_date=d, sofr=_sofr(d), dividends={})

        self.loop = ComputeLoop(
            self.cfg,
            make_subscriber=lambda: None,  # type: ignore[arg-type,return-value]
            publish=lambda t, p: self.published.append((t, p)),
            publisher_ready=threading.Event(),
            inputs=InputsHolder(loader),
            cold=cold if cold is not None else FakeCold(),
            on_fatal=self.fatal.append,
            now=lambda: self.now_dt,
        )

    def opt_quote(self, topic: str, osi: str, bid: float, ask: float, ts: int) -> None:
        self.loop.step(topic, json.dumps({"type": "quote", "symbol": osi, "bid": bid, "ask": ask, "timestamp": ts}))

    def opt_trade(self, topic: str, osi: str, price: float, ts: int) -> None:
        self.loop.step(topic, json.dumps({"type": "trade", "symbol": osi, "price": price, "timestamp": ts}))

    def eq_quote(self, sym: str, bid: float, ask: float, ts: int) -> None:
        self.loop.step(f"md.equity.quote.{sym}", json.dumps({"type": "quote", "symbol": sym, "bid": bid, "ask": ask, "timestamp": ts}))

    def eq_trade(self, sym: str, price: float, ts: int) -> None:
        self.loop.step(f"md.equity.trade.{sym}", json.dumps({"type": "trade", "symbol": sym, "price": price, "timestamp": ts}))

    def time_tick(self, ts: int) -> None:
        self.loop.step(TopicBuilder.for_options_time(), json.dumps({"type": "time", "ts": ts}))

    def last(self) -> dict[str, Any]:
        return self.published[-1][1]


class FakeCold:
    """Stands in for ColdPath. `price` returns a pre-seeded ColdResult per osi; `stock_prices`
    answers from a seeded (symbol, ts) -> SelectedPrice map, recording its calls."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.by_osi: dict[str, Any] = {}
        self.stock: dict[tuple[str, int], SelectedPrice] = {}
        self.stock_calls: list[list[tuple[str, int]]] = []

    def price(self, osis: list[str], stock_at: Any = None) -> list[Any]:
        self.calls.append(list(osis))
        return [self.by_osi[o] for o in osis]

    def stock_prices(self, pairs: Any) -> dict[tuple[str, int], SelectedPrice]:
        pairs = list(pairs)
        self.stock_calls.append(pairs)
        return {p: self.stock[p] for p in pairs if p in self.stock}


def test_option_tick_prices_as_of_its_timestamp_from_the_stock_history() -> None:
    h = Harness()
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))          # history only
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0, 5))    # reprices now, as of its own ts
    assert [t for t, _ in h.published] == ["shadow.md.option.greeks.NVDA.2026-11-20.C.180_00"]
    msg = h.last()
    assert msg["status"] == "success"
    assert msg["priced_as_of"] == et_ms(11, 0, 5)            # the option quote's timestamp
    assert msg["spot"] == pytest.approx(181.01)              # the stock mid from history, at or before T
    assert msg["spot_ts"] == et_ms(11, 0)


def test_a_stock_tick_does_not_reprice() -> None:
    h = Harness()
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0, 5))
    n = len(h.published)
    h.eq_quote("NVDA", 181.5, 181.52, et_ms(11, 0, 10))      # later stock: history only, no publish
    h.eq_trade("NVDA", 181.7, et_ms(11, 0, 11))
    assert len(h.published) == n


def test_time_event_reprices_every_contract() -> None:
    h = Harness()
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0, 1))
    h.opt_quote(OPT_P, NVDA_P, 3.0, 3.2, et_ms(11, 0, 1))
    h.published.clear()
    h.eq_quote("NVDA", 182.0, 182.02, et_ms(11, 0, 4))       # moves the stock, does not reprice
    h.time_tick(et_ms(11, 0, 5))                             # one full pass as of 11:00:05
    topics = sorted(t for t, _ in h.published)
    assert topics == [
        "shadow.md.option.greeks.NVDA.2026-11-20.C.180_00",
        "shadow.md.option.greeks.NVDA.2026-11-20.P.170_00",
    ]
    for _, msg in h.published:
        assert msg["priced_as_of"] == et_ms(11, 0, 5)
        assert msg["spot"] == pytest.approx(182.01)          # the 11:00:04 stock, now reflected


def test_an_older_result_never_replaces_a_newer_one() -> None:
    h = Harness()
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0, 1))
    h.time_tick(et_ms(12, 0))                                # newest result priced as of 12:00
    h.published.clear()
    h.time_tick(et_ms(11, 30))                               # older: rejected, nothing published
    assert h.published == []


def test_option_tick_with_no_history_enqueues_instead_of_publishing_a_failure() -> None:
    h = Harness()
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(17, 0))       # after hours, no stock history
    assert h.published == []                                 # no missing_spot failure published
    assert h.loop._cold_queue.get_nowait() == NVDA_C         # handed to the resolver instead


def test_resolver_prices_a_history_miss_from_mds_stock_not_a_failure() -> None:
    # A bus quote (10:00) older than the stock history (11:00) is a miss. The resolver fetches the
    # stock from MDS at the quote's own time and publishes a success — not a missing_spot failure.
    h = Harness()
    cold = FakeCold()
    cold.stock = {("NVDA", et_ms(10, 0)): SelectedPrice(180.0, PriceKind.LAST, et_ms(10, 0))}
    h.loop._cold = cold  # type: ignore[assignment]
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))          # history starts at 11:00
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(10, 0))       # older than history -> miss -> enqueue
    assert h.published == []
    osi = h.loop._cold_queue.get_nowait()
    h.loop._price_with_mds_stock([osi])                      # what the resolver thread runs
    assert cold.stock_calls == [[("NVDA", et_ms(10, 0))]]    # MDS asked at the quote's time
    msg = h.last()
    assert msg["status"] == "success" and msg["priced_as_of"] == et_ms(10, 0)
    assert msg["spot"] == pytest.approx(180.0) and "reason" not in msg


def test_get_greeks_reprices_a_held_failure_through_the_cold_path() -> None:
    h = Harness()
    cold = FakeCold()
    t = et_ms(17, 0)
    cold.stock = {("NVDA", t): SelectedPrice(186.64, PriceKind.LAST, t)}
    h.loop._cold = cold  # type: ignore[assignment]
    # Seed a held missing_spot failure (as the after-hours lease bootstrap would have produced) with
    # a held option quote so the option price is recoverable.
    h.loop.book.ensure(NVDA_C)
    h.loop.book.on_option_quote(NVDA_C, Quote(9.0, 9.2, t))
    h.loop.book.accept(OptionGreeksMessage(
        osi=NVDA_C, status=RowStatus.FAILURE, priced_as_of=t, methodology_version="v2",
        reason=FailureReason.MISSING_SPOT))

    out = h.loop.get_greeks([NVDA_C])
    assert out[0].status.value == "success"                  # re-priced, not the held failure
    assert out[0].spot == pytest.approx(186.64)
    assert cold.stock_calls == [[("NVDA", t)]]


def test_expired_contract_dropped_and_its_history_pruned() -> None:
    h = Harness()
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0, 1))
    assert NVDA_C in h.loop.book.contracts
    # A time tick after the 2026-11-20 16:00 ET close: the contract is dropped before the pass and
    # its underlying's history is pruned (nothing left in the book references NVDA).
    h.time_tick(et_ms(16, 1, 0, day=date(2026, 11, 20)))
    assert NVDA_C not in h.loop.book.contracts
    assert h.loop.stock.at("NVDA", et_ms(16, 1, 0, day=date(2026, 11, 20))) is None


def test_get_greeks_answers_from_book_then_cold_path() -> None:
    # Real ColdPath with a fake MDS transport: the second contract is not in the book, and its quote
    # (10:00) predates the stock history (11:00), so the stock is fetched from MDS at the quote's ts.
    requests: list[dict[str, Any]] = []
    qts = et_ms(10, 0, 0)

    def transport(req: dict[str, Any]) -> dict[str, Any]:
        requests.append(req)
        if req["op"] == "get_option_snapshots":
            snaps = {NVDA_P: {"osi": NVDA_P, "quote": {"bid": 3.0, "ask": 3.2, "at": qts}}}
            return {"ok": True, "data": {"snapshots": snaps, "count": 1}}
        return {"ok": True, "data": {"prices": [{"symbol": "NVDA", "as_of": qts, "last": 181.0, "trade_ts": qts - 1000}]}}

    h = Harness()
    cold = ColdPath(transport, h.loop._inputs, Methodology("v2", 15.0, "16:00"))
    h.loop._cold = cold  # type: ignore[assignment]

    # Seed the book with one contract via a live tick (history for NVDA starts at 11:00).
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0, 1))
    h.published.clear()

    out = h.loop.get_greeks([NVDA_C, NVDA_P])
    assert [m.osi for m in out] == [NVDA_C, NVDA_P]
    assert out[0].priced_as_of == et_ms(11, 0, 1)            # from the book
    assert out[1].status.value == "success" and out[1].priced_as_of == qts  # cold-path, as of the quote
    # The stock was asked for at the quote's own timestamp (history did not reach back to 10:00).
    eq_req = next(r for r in requests if r["op"] == "get_equity_prices_at")
    assert eq_req["symbols_at"] == [{"symbol": "NVDA", "ts": qts}]
    # The cold-path result was added to the book and published.
    assert NVDA_P in h.loop.book.contracts
    assert any(t.endswith("P.170_00") for t, _ in h.published)


def test_cold_path_uses_the_stock_history_and_skips_mds_when_it_reaches_back() -> None:
    # The cold contract's quote (11:00:02) is covered by the stock history (NVDA quote at 11:00), so
    # the cold path prices from the history and never calls get_equity_prices_at.
    requests: list[dict[str, Any]] = []
    qts = et_ms(11, 0, 2)

    def transport(req: dict[str, Any]) -> dict[str, Any]:
        requests.append(req)
        if req["op"] == "get_option_snapshots":
            return {"ok": True, "data": {"snapshots": {NVDA_P: {"osi": NVDA_P, "quote": {"bid": 3.0, "ask": 3.2, "at": qts}}}, "count": 1}}
        raise AssertionError("get_equity_prices_at must not be called when the history covers T")

    h = Harness()
    h.loop._cold = ColdPath(transport, h.loop._inputs, Methodology("v2", 15.0, "16:00"))  # type: ignore[assignment]
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))          # history for NVDA at 11:00

    out = h.loop.get_greeks([NVDA_P])
    assert out[0].status.value == "success" and out[0].spot == pytest.approx(181.01)
    assert [r["op"] for r in requests] == ["get_option_snapshots"]   # no equity fetch


def test_friday_curb_quote_priced_against_the_stock_at_that_time() -> None:
    h = Harness()
    h.now_dt = datetime(2026, 11, 13, 22, 0, tzinfo=timezone.utc)  # ~17:00 ET, after the curb
    h.eq_trade("NVDA", 186.64, et_ms(16, 0))                 # the 16:00 close
    h.eq_trade("NVDA", 186.90, et_ms(16, 5))                 # a 16:05 curb print
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(16, 5))       # re-quoted at 16:05
    msg = h.last()
    assert msg["status"] == "success"
    assert msg["priced_as_of"] == et_ms(16, 5)
    assert msg["spot"] == pytest.approx(186.90) and msg["spot_ts"] == et_ms(16, 5)


def test_failed_input_load_is_fatal_through_run() -> None:
    h = Harness(loader_fails=True)
    with pytest.raises(ValueError):
        h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0))   # for_date raises before the lock


def test_run_reports_fatal_and_returns() -> None:
    h = Harness(loader_fails=True)

    class Sub:
        def __init__(self) -> None:
            self._sent = False

        def recv(self) -> tuple[str, str]:
            if not self._sent:
                self._sent = True
                return OPT_C, json.dumps({"type": "quote", "symbol": NVDA_C, "bid": 9.0, "ask": 9.2, "timestamp": et_ms(11, 0)})
            return "", ""

        def stop(self) -> None: ...

        def close(self) -> None: ...

    ready = threading.Event()
    ready.set()
    h.loop._make_sub = lambda: Sub()  # type: ignore[assignment]
    h.loop._ready = ready
    h.loop.run()
    assert len(h.fatal) == 1 and isinstance(h.fatal[0], ValueError)


def test_thread_start_stop_join() -> None:
    h = Harness()
    h.loop.start()
    h.loop.stop()
    h.loop.join(timeout=2.0)
    assert not h.loop.is_alive()
