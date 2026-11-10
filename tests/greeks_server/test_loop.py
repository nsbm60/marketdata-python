"""ComputeLoop.step() and get_greeks under the live-bus-only design (greeks-server-spec.md §3-§6).

Driven directly through step() — no sockets, no threads. The bus carries live ticks only, so a live
option tick pairs with the held live stock; a contract with no held stock is fetched from MDS; the
open (a new session_open_ts) is a restart. Includes the scenario tests run before deploy.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from datetime import date, datetime, time as dtime, timezone
from typing import Any
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
DAY = date(2026, 11, 13)          # a Friday, before the 2026-11-20 expiry; EST (UTC-5)
NVDA_C = "NVDA261120C00180000"
NVDA_P = "NVDA261120P00170000"
OPT_C = "md.option.quote.NVDA.2026-11-20.C.180_00"
OPT_P = "md.option.quote.NVDA.2026-11-20.P.170_00"


def et_ms(h: int, m: int, s: int = 0, day: date = DAY) -> int:
    return int(datetime.combine(day, dtime(h, m, s), tzinfo=ET).astimezone(timezone.utc).timestamp() * 1000)


SESSION_OPEN = et_ms(9, 30)       # the DAY options-session open


def _sofr(d: date) -> dict[date, float]:
    from datetime import timedelta
    return {d - timedelta(days=i): 0.043 for i in range(1, 20)}


class Harness:
    def __init__(self, cold: Any = None, loader_fails: bool = False) -> None:
        self.cfg = replace(load_config(), server=replace(load_config().server, topic_prefix="shadow."))
        self.published: list[tuple[str, dict[str, Any]]] = []
        self.loads: list[date] = []
        self.fatal: list[BaseException] = []
        self.now_dt = datetime(2026, 11, 13, 23, 59, tzinfo=timezone.utc)  # late: no tick is "future"

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

    def time_tick(self, ts: int, session_open: int | None = SESSION_OPEN) -> None:
        payload: dict[str, Any] = {"type": "time", "ts": ts}
        if session_open is not None:
            payload["session_open_ts"] = session_open
        self.loop.step(TopicBuilder.for_options_time(), json.dumps(payload))

    def drain_resolver(self) -> None:
        """Run what the resolver thread would: price every queued no-stock option tick from MDS."""
        batch: list[str] = []
        while not self.loop._cold_queue.empty():
            osi = self.loop._cold_queue.get_nowait()
            if osi is not None:
                batch.append(osi)
        if batch:
            self.loop._price_with_mds_stock(batch)

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

    def price(self, osis: list[str]) -> list[Any]:
        self.calls.append(list(osis))
        return [self.by_osi[o] for o in osis]

    def stock_prices(self, pairs: Any) -> dict[tuple[str, int], SelectedPrice]:
        pairs = list(pairs)
        self.stock_calls.append(pairs)
        return {p: self.stock[p] for p in pairs if p in self.stock}


# A transport (for the real ColdPath) that serves get_option_snapshots and get_equity_prices_at from
# seeded dicts. Lets scenario tests drive the cold path end to end with no network.
def cold_transport(snapshots: dict[str, dict[str, Any]], stock: dict[tuple[str, int], tuple[float, int]],
                   log: list[dict[str, Any]] | None = None, session_open: int = 0):
    def transport(req: dict[str, Any]) -> dict[str, Any]:
        if log is not None:
            log.append(req)
        if req["op"] == "get_option_snapshots":
            want = set(req["symbols"])
            snaps = {o: s for o, s in snapshots.items() if o in want}
            return {"ok": True, "data": {"snapshots": snaps, "count": len(snaps), "session_open_ts": session_open}}
        prices = []
        for item in req["symbols_at"]:
            key = (item["symbol"], item["ts"])
            if key in stock:
                last, trade_ts = stock[key]
                prices.append({"symbol": item["symbol"], "as_of": item["ts"], "last": last, "trade_ts": trade_ts})
        return {"ok": True, "data": {"prices": prices}}
    return transport


# ── live route ────────────────────────────────────────────────────────────────────────────────

def test_live_option_tick_pairs_with_the_held_live_stock() -> None:
    h = Harness()
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))          # held live stock
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0, 5))    # live option tick
    assert [t for t, _ in h.published] == ["shadow.md.option.greeks.NVDA.2026-11-20.C.180_00"]
    msg = h.last()
    assert msg["status"] == "success"
    assert msg["priced_as_of"] == et_ms(11, 0, 5)
    assert msg["spot"] == pytest.approx(181.01)              # the held mid


def test_a_stock_tick_does_not_reprice() -> None:
    h = Harness()
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0, 5))
    n = len(h.published)
    h.eq_quote("NVDA", 182.0, 182.02, et_ms(11, 0, 10))      # only updates held stock
    h.eq_trade("NVDA", 182.1, et_ms(11, 0, 11))
    assert len(h.published) == n


def test_a_live_option_tick_with_no_held_stock_is_fetched_not_failed() -> None:
    h = Harness()
    cold = FakeCold()
    cold.stock = {("NVDA", et_ms(17, 0)): SelectedPrice(186.64, PriceKind.LAST, et_ms(17, 0))}
    h.loop._cold = cold  # type: ignore[assignment]
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(17, 0))       # after hours, no equity ticks held
    assert h.published == []                                 # no missing_spot failure
    assert h.loop._cold_queue.get_nowait() == NVDA_C         # handed to the resolver
    h.loop._cold_queue.put(NVDA_C)                            # (put it back for drain)
    h.drain_resolver()
    msg = h.last()
    assert msg["status"] == "success" and msg["spot"] == pytest.approx(186.64)
    assert cold.stock_calls == [[("NVDA", et_ms(17, 0))]]


# ── time route + the open as a restart ───────────────────────────────────────────────────────

def test_time_event_reprices_the_book() -> None:
    h = Harness()
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0, 1))
    h.opt_quote(OPT_P, NVDA_P, 3.0, 3.2, et_ms(11, 0, 1))
    h.published.clear()
    h.eq_quote("NVDA", 182.0, 182.02, et_ms(11, 0, 4))       # moves held stock, no reprice
    h.time_tick(et_ms(11, 0, 5))
    assert sorted(t for t, _ in h.published) == [
        "shadow.md.option.greeks.NVDA.2026-11-20.C.180_00",
        "shadow.md.option.greeks.NVDA.2026-11-20.P.170_00",
    ]
    for _, msg in h.published:
        assert msg["priced_as_of"] == et_ms(11, 0, 5) and msg["spot"] == pytest.approx(182.01)


def test_the_open_is_a_restart() -> None:
    h = Harness()
    # Yesterday's state: a time event with yesterday's open, a contract, held stock.
    yday_open = et_ms(9, 30, day=date(2026, 11, 12))
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0, day=date(2026, 11, 12)))
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0, 1, day=date(2026, 11, 12)))
    h.time_tick(et_ms(12, 0, day=date(2026, 11, 12)), session_open=yday_open)
    assert NVDA_C in h.loop.book.contracts
    h.published.clear()

    # Today's first tick carries a new session_open_ts: the book and held stock are cleared.
    h.time_tick(et_ms(9, 30, 1), session_open=SESSION_OPEN)
    assert h.loop.book.contracts == {}                       # nothing from yesterday survives
    assert not h.loop.stock.has("NVDA")
    assert h.published == []                                  # an empty book reprices nothing

    # A fresh live tick prices against today's stock.
    h.eq_quote("NVDA", 190.0, 190.02, et_ms(9, 31))
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(9, 31, 1))
    assert h.last()["spot"] == pytest.approx(190.01)


def test_midnight_into_premarket_clears_nothing() -> None:
    h = Harness()
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0, 1))
    h.time_tick(et_ms(12, 0))                                 # sets session_open = DAY open
    # No time event between the close and the next open: the book persists through pre-market.
    assert NVDA_C in h.loop.book.contracts
    h.published.clear()
    # A pre-market live tick (next day) prices live against the pre-market stock.
    pm = date(2026, 11, 16)                                   # Monday
    h.now_dt = datetime(2026, 11, 16, 13, 0, tzinfo=timezone.utc)
    h.eq_quote("NVDA", 184.0, 184.02, et_ms(8, 0, day=pm))
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(8, 0, 1, day=pm))
    assert h.last()["status"] == "success" and h.last()["spot"] == pytest.approx(184.01)
    assert NVDA_C in h.loop.book.contracts                    # still not cleared (no open tick yet)


def test_expired_contract_dropped_on_a_time_event() -> None:
    h = Harness()
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0, 1))
    assert NVDA_C in h.loop.book.contracts
    expiry_day = date(2026, 11, 20)
    h.time_tick(et_ms(16, 1, 0, day=expiry_day), session_open=et_ms(9, 30, day=expiry_day))
    assert NVDA_C not in h.loop.book.contracts


def test_older_result_never_replaces_a_newer_one() -> None:
    h = Harness()
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0, 1))
    h.time_tick(et_ms(12, 0))
    h.published.clear()
    h.time_tick(et_ms(11, 30))                                # older: rejected
    assert h.published == []


# ── 16:05 Curb ────────────────────────────────────────────────────────────────────────────────

def test_curb_tick_at_1605_prices_live_against_the_stock_at_that_moment() -> None:
    h = Harness()
    h.now_dt = datetime(2026, 11, 13, 22, 0, tzinfo=timezone.utc)
    h.eq_trade("NVDA", 186.64, et_ms(16, 0))
    h.eq_trade("NVDA", 186.90, et_ms(16, 5))                  # the 16:05 curb stock
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(16, 5))
    msg = h.last()
    assert msg["status"] == "success" and msg["priced_as_of"] == et_ms(16, 5)
    assert msg["spot"] == pytest.approx(186.90)


# ── get_greeks ──────────────────────────────────────────────────────────────────────────────────

def test_get_greeks_returns_held_success_and_cold_paths_the_rest() -> None:
    log: list[dict[str, Any]] = []
    qts = et_ms(11, 0)
    snaps = {NVDA_P: {"osi": NVDA_P, "quote": {"bid": 3.0, "ask": 3.2, "at": qts}}}
    stock = {("NVDA", qts): (181.0, qts - 1000)}
    h = Harness()
    h.loop._cold = ColdPath(cold_transport(snaps, stock, log), h.loop._inputs, Methodology("v2", 15.0, "16:00"))  # type: ignore[assignment]
    # Seed one held success via a live tick.
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))
    h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0, 1))
    h.published.clear()

    out = h.loop.get_greeks([NVDA_C, NVDA_P])
    assert [m.osi for m in out] == [NVDA_C, NVDA_P]
    assert out[0].priced_as_of == et_ms(11, 0, 1)             # from the book
    assert out[1].status.value == "success" and out[1].priced_as_of == qts  # cold path
    assert any(r["op"] == "get_equity_prices_at" and r["symbols_at"] == [{"symbol": "NVDA", "ts": qts}] for r in log)
    assert NVDA_P in h.loop.book.contracts


def test_get_greeks_reprices_a_held_failure_through_the_cold_path() -> None:
    log: list[dict[str, Any]] = []
    t = et_ms(17, 0)
    snaps = {NVDA_C: {"osi": NVDA_C, "quote": {"bid": 9.0, "ask": 9.2, "at": t}}}
    stock = {("NVDA", t): (186.64, t)}
    h = Harness()
    h.loop._cold = ColdPath(cold_transport(snaps, stock, log), h.loop._inputs, Methodology("v2", 15.0, "16:00"))  # type: ignore[assignment]
    # Seed a held missing_spot failure.
    h.loop.book.ensure(NVDA_C)
    h.loop.book.accept(OptionGreeksMessage(
        osi=NVDA_C, status=RowStatus.FAILURE, priced_as_of=t, methodology_version="v2",
        reason=FailureReason.MISSING_SPOT))
    out = h.loop.get_greeks([NVDA_C])
    assert out[0].status.value == "success" and out[0].spot == pytest.approx(186.64)


# ── scenarios ─────────────────────────────────────────────────────────────────────────────────

def test_stream_ignores_a_trade_only_option_tick() -> None:
    # Only quotes price greeks; a trade-only option tick publishes nothing (no trade fallback).
    h = Harness()
    h.eq_quote("NVDA", 181.0, 181.02, et_ms(11, 0))
    h.opt_trade(OPT_C, NVDA_C, 9.1, et_ms(11, 0, 5))
    assert h.published == [] and NVDA_C not in h.loop.book.contracts


def test_cold_path_rejects_a_quote_before_session_open() -> None:
    # A snapshot quote from before the session open is no_current_quote, with no stock lookup; one
    # after it prices.
    log: list[dict[str, Any]] = []
    stale, fresh = "NVDA261120C00300000", NVDA_C
    before, after = SESSION_OPEN - 60_000, SESSION_OPEN + 60_000
    snaps = {
        stale: {"osi": stale, "quote": {"bid": 0.1, "ask": 0.2, "at": before}},
        fresh: {"osi": fresh, "quote": {"bid": 9.0, "ask": 9.2, "at": after}},
    }
    stock = {("NVDA", after): (181.0, after)}
    h = Harness()
    h.loop._cold = ColdPath(cold_transport(snaps, stock, log, session_open=SESSION_OPEN),
                            h.loop._inputs, Methodology("v2", 15.0, "16:00"))  # type: ignore[assignment]
    out = {m.osi: m for m in h.loop.get_greeks([stale, fresh])}
    assert out[stale].reason is FailureReason.NO_CURRENT_QUOTE and out[stale].status is RowStatus.FAILURE
    assert out[fresh].status.value == "success" and out[fresh].spot == pytest.approx(181.0)
    # No stock was asked for the stale contract's time; only the fresh one's.
    eq = next(r for r in log if r["op"] == "get_equity_prices_at")
    assert eq["symbols_at"] == [{"symbol": "NVDA", "ts": after}]


def test_cold_path_after_hours_prices_1600_and_curb_against_stock_at_each_time() -> None:
    log: list[dict[str, Any]] = []
    close_osi, curb_osi = NVDA_C, NVDA_P
    t1600, t1605 = et_ms(16, 0), et_ms(16, 5)
    snaps = {
        close_osi: {"osi": close_osi, "quote": {"bid": 9.0, "ask": 9.2, "at": t1600}},
        curb_osi: {"osi": curb_osi, "quote": {"bid": 3.0, "ask": 3.2, "at": t1605}},
    }
    stock = {("NVDA", t1600): (186.64, t1600), ("NVDA", t1605): (186.90, t1605)}
    h = Harness()
    h.loop._cold = ColdPath(cold_transport(snaps, stock, log, session_open=SESSION_OPEN),
                            h.loop._inputs, Methodology("v2", 15.0, "16:00"))  # type: ignore[assignment]
    out = {m.osi: m for m in h.loop.get_greeks([close_osi, curb_osi])}
    assert out[close_osi].status.value == "success" and out[close_osi].spot == pytest.approx(186.64)
    assert out[close_osi].priced_as_of == t1600
    assert out[curb_osi].status.value == "success" and out[curb_osi].spot == pytest.approx(186.90)
    assert out[curb_osi].priced_as_of == t1605


def test_scenario_new_ticker_opened_after_the_close() -> None:
    # RKLB chain, never seen, opened after the close: greeks from the cold path, priced as of each
    # contract's last quote with the stock at that moment.
    log: list[dict[str, Any]] = []
    rklb_c, rklb_p = "RKLB261120C00030000", "RKLB261120P00025000"
    q1, q2 = et_ms(16, 30), et_ms(16, 31)
    snaps = {
        rklb_c: {"osi": rklb_c, "quote": {"bid": 1.0, "ask": 1.1, "at": q1}},
        rklb_p: {"osi": rklb_p, "quote": {"bid": 0.5, "ask": 0.6, "at": q2}},
    }
    stock = {("RKLB", q1): (31.0, q1), ("RKLB", q2): (31.0, q2)}
    h = Harness()
    h.loop._cold = ColdPath(cold_transport(snaps, stock, log), h.loop._inputs, Methodology("v2", 15.0, "16:00"))  # type: ignore[assignment]
    out = h.loop.get_greeks([rklb_c, rklb_p])
    assert all(m.status.value == "success" for m in out)
    assert out[0].priced_as_of == q1 and out[1].priced_as_of == q2   # as of each last quote


def test_scenario_restart_then_release_completes_well_inside_the_ping_window() -> None:
    # A greeks-server restart, then CalcServer re-leases hundreds of held contracts: all priced
    # through the cold path. Measure the compute wall time (fakes, so this is solver + bookkeeping,
    # not network) and assert it is well inside the self-ping health window (interval x max_missed).
    import math
    unders = ["NVDA", "MU", "RKLB", "AMD", "AAPL"]
    osis, snaps, stock = [], {}, {}
    base = et_ms(10, 0)
    for u in unders:
        for i in range(40):                                   # 200 contracts across 5 underlyings
            strike = 100000 + i * 1000
            osi = f"{u}261120C{strike:08d}"
            osis.append(osi)
            ts = base + i
            snaps[osi] = {"osi": osi, "quote": {"bid": 5.0, "ask": 5.2, "at": ts}}
            stock[(u, ts)] = (120.0, ts)
    h = Harness()
    h.loop._cold = ColdPath(cold_transport(snaps, stock), h.loop._inputs, Methodology("v2", 15.0, "16:00"))  # type: ignore[assignment]

    t0 = time.perf_counter()
    out = h.loop.get_greeks(osis)
    elapsed = time.perf_counter() - t0

    assert len(out) == len(osis)
    # Self-ping health window = SELF_PING_INTERVAL_SEC * SELF_PING_MAX_MISSED. With concurrent
    # dispatch pings are answered inline regardless, so any sane compute time is safe; assert a wide
    # margin and report the measurement.
    from service.constants import SELF_PING_INTERVAL_SEC, SELF_PING_MAX_MISSED
    window = SELF_PING_INTERVAL_SEC * SELF_PING_MAX_MISSED
    print(f"\n[scenario] re-lease of {len(osis)} contracts cold-pathed in {elapsed*1000:.0f} ms "
          f"(self-ping window {window}s)")
    assert elapsed < window
    assert not math.isnan(elapsed)


def test_failed_input_load_is_fatal_through_run() -> None:
    h = Harness(loader_fails=True)
    with pytest.raises(ValueError):
        h.opt_quote(OPT_C, NVDA_C, 9.0, 9.2, et_ms(11, 0))


def test_thread_start_stop_join() -> None:
    h = Harness()
    h.loop.start()
    h.loop.stop()
    h.loop.join(timeout=2.0)
    assert not h.loop.is_alive()
