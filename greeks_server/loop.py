"""The compute loop: one thread, one subscriber, driven by data (greeks-server-spec.md §3-§6).

There is no session and no clock in the pricing. Each message off the bus does one thing:

  - an option quote or trade  -> reprice that one contract as of the tick's own timestamp, now;
  - an equity quote or trade  -> add the price to the underlying's history, nothing else. Off-hours
    the stock moves while option quotes stand still, so a stock tick must not pair a stale option
    quote with a later stock price; in session, repricing every strike on every stock tick is heavy
    and mostly recomputes greeks that barely moved. The 5-second pass brings quiet strikes up to
    date instead;
  - a cal.options.time tick (ts) -> one full pass: reprice every contract in the book as of ts, from
    its held option price and the stock at or before ts. One steady pass per 5 seconds.

The book and the stock history are shared with get_greeks (served on the router thread), so a lock
guards them. The book is never cleared; a contract is dropped once its expiry's close has passed.

A failure to load the rate/dividends for a date is fatal: on_fatal stops the service and systemd
restarts it. Per-contract solver errors are isolated (compute_or_fail) and published as failures.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, time as dtime, timezone
from typing import Any, Callable, Optional, Protocol
from zoneinfo import ZoneInfo

from greeks.config import GreeksConfig
from greeks.topic_builder import TopicBuilder
from greeks_server.book import Book
from greeks_server.coldpath import ColdPath
from greeks_server.compute import Contract, Methodology, compute_or_fail
from greeks_server.inputs import InputsHolder
from greeks_server.message import OptionGreeksMessage
from greeks_server.price import SelectedPrice, select_price
from greeks_server.stock_history import StockHistory
from greeks_server.wire import Tick, decode_tick

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
STATS_INTERVAL_S = 60.0


class Subscriber(Protocol):
    def recv(self) -> tuple[str, str]: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


def methodology(cfg: GreeksConfig) -> Methodology:
    return Methodology(
        methodology_version=cfg.methodology_version,
        t_floor_minutes=float(cfg.t_floor_minutes),
        expiry_time_et=cfg.expiry_time_et,
    )


def greeks_topic(prefix: str, c: Contract) -> str:
    """prefix + md.option.greeks.<UND>.<YYYY-MM-DD>.<C|P>.<d>_<cc> (TopicBuilder; OSI strike is mills)."""
    return prefix + TopicBuilder.for_option_greeks(c.underlying, c.expiry, c.side, f"{c.strike:.3f}")


class ComputeLoop(threading.Thread):
    def __init__(
        self,
        cfg: GreeksConfig,
        make_subscriber: Callable[[], Subscriber],
        publish: Callable[[str, dict[str, Any]], None],
        publisher_ready: threading.Event,
        inputs: InputsHolder,
        cold: ColdPath,
        on_fatal: Callable[[BaseException], None],
        *,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(name="greeks-compute", daemon=True)
        self._cfg = cfg
        self._m = methodology(cfg)
        self._prefix = cfg.server.topic_prefix
        self._make_sub = make_subscriber
        self._publish = publish
        self._ready = publisher_ready
        self._inputs = inputs
        self._cold = cold
        self._on_fatal = on_fatal
        self._now = now
        self._mono = monotonic
        self._stop_event = threading.Event()
        self._sub: Optional[Subscriber] = None

        self._lock = threading.Lock()
        self.book = Book()
        self.stock = StockHistory()
        self._stats = _Stats()
        self._time_topic = TopicBuilder.for_options_time()

    # ── lifecycle ─────────────────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_event.set()
        sub = self._sub
        if sub is not None:
            sub.stop()

    def run(self) -> None:
        try:
            while not self._ready.wait(0.25):
                if self._stop_event.is_set():
                    return
            sub = self._make_sub()
            self._sub = sub
            try:
                while not self._stop_event.is_set():
                    topic, payload = sub.recv()
                    self.step(topic, payload)
            finally:
                sub.close()
        except BaseException as e:  # noqa: BLE001 — anything here ends the process, loudly
            log.exception("compute loop failed: %s", e)
            self._on_fatal(e)

    # ── one message (public for tests) ──────────────────────────────────────────────────────────

    def step(self, topic: str, payload: str) -> None:
        if topic:
            if topic == self._time_topic:
                ts = _time_ts(payload)
                if ts is not None:
                    self._on_time(ts)
                else:
                    self._stats.bad_ticks += 1
            elif topic.startswith(TopicBuilder.prefix_option_quotes()) or topic.startswith(
                TopicBuilder.prefix_option_trades()
            ):
                self._on_option(payload)
            elif topic.startswith(TopicBuilder.prefix_equity_quotes()) or topic.startswith(
                TopicBuilder.prefix_equity_trades()
            ):
                self._on_equity(payload)
        self._stats.maybe_log(self._mono())

    # ── triggers ──────────────────────────────────────────────────────────────────────────────

    def _on_option(self, payload: str) -> None:
        tick = self._decode(payload)
        if tick is None:
            return
        osi, t_ms = tick.symbol, _tick_ts(tick)
        as_of = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
        inputs = self._inputs.for_date(as_of.astimezone(ET).date())  # outside the lock; raises -> fatal
        with self._lock:
            contract = self.book.ensure(osi)
            if contract is None:
                return
            changed = False
            if tick.quote is not None:
                changed = self.book.on_option_quote(osi, tick.quote)
            if tick.trade is not None:
                changed = self.book.on_option_trade(osi, tick.trade) or changed
            if not changed:
                return  # an older or duplicate tick: the held price and its result stand
            opt_px = select_price(self.book.option_quotes.get(osi), self.book.option_trades.get(osi), t_ms)
            spot_px = self.stock.at(contract.underlying, t_ms)
            msg = compute_or_fail(contract, opt_px, spot_px, inputs, as_of, self._m)
            accepted = self.book.accept(msg)
            topic = greeks_topic(self._prefix, contract)
        if accepted:
            self._publish(topic, msg.to_wire())
            self._stats.reprices += 1

    def _on_equity(self, payload: str) -> None:
        tick = self._decode(payload)
        if tick is None:
            return
        with self._lock:
            if tick.quote is not None:
                self.stock.add_quote(tick.symbol, tick.quote.ts_ms, tick.quote.bid, tick.quote.ask)
            if tick.trade is not None:
                self.stock.add_trade(tick.symbol, tick.trade.ts_ms, tick.trade.price)

    def _on_time(self, ts_ms: int) -> None:
        as_of = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        inputs = self._inputs.for_date(as_of.astimezone(ET).date())  # outside the lock; raises -> fatal
        t0 = time.perf_counter()
        published: list[tuple[str, dict[str, Any]]] = []
        ok = 0
        with self._lock:
            self._drop_expired(ts_ms)
            self.stock.drop_symbols(self.book.underlyings())
            contracts = list(self.book.contracts.values())
            for c in contracts:
                opt_px = select_price(self.book.option_quotes.get(c.osi), self.book.option_trades.get(c.osi), ts_ms)
                spot_px = self.stock.at(c.underlying, ts_ms)
                msg = compute_or_fail(c, opt_px, spot_px, inputs, as_of, self._m)
                if self.book.accept(msg):
                    published.append((greeks_topic(self._prefix, c), msg.to_wire()))
                    ok += msg.reason is None
        for topic, wire in published:
            self._publish(topic, wire)
        self._stats.record(len(contracts), ok, time.perf_counter() - t0)

    def _drop_expired(self, now_ms: int) -> None:
        """Drop any contract whose expiry date's close is already behind ``now_ms`` (held under lock)."""
        gone = [osi for osi, c in self.book.contracts.items() if self._expiry_close_ms(c) < now_ms]
        for osi in gone:
            self.book.drop(osi)
        if gone:
            log.info("dropped %d expired contract(s): %s", len(gone), ", ".join(sorted(gone)[:8]))

    def _expiry_close_ms(self, c: Contract) -> int:
        hh, mm = (int(x) for x in self._m.expiry_time_et.split(":"))
        close = datetime.combine(c.expiry, dtime(hh, mm), tzinfo=ET).astimezone(timezone.utc)
        return int(close.timestamp() * 1000)

    # ── get_greeks (router thread) ──────────────────────────────────────────────────────────────

    def get_greeks(self, osis: list[str]) -> list[OptionGreeksMessage]:
        """Answer each osi from the book; price any the book has never seen through the cold path,
        add them to the book, and publish them. One message per osi, in order."""
        results: list[Optional[OptionGreeksMessage]] = [None] * len(osis)
        missing: list[str] = []
        with self._lock:
            for i, osi in enumerate(osis):
                held = self.book.results.get(osi)
                if held is not None:
                    results[i] = held
                else:
                    missing.append(osi)
        if missing:
            cold = self._cold.price(missing, self._history_stock_at)  # MDS I/O, no lock
            by_osi: dict[str, OptionGreeksMessage] = {cr.osi: cr.message for cr in cold}
            publish: list[tuple[str, dict[str, Any]]] = []
            with self._lock:
                for cr in cold:
                    contract = self.book.ensure(cr.osi)
                    if contract is None:
                        continue
                    if cr.quote is not None:
                        self.book.on_option_quote(cr.osi, cr.quote)
                    if cr.trade is not None:
                        self.book.on_option_trade(cr.osi, cr.trade)
                    if self.book.accept(cr.message):
                        publish.append((greeks_topic(self._prefix, contract), cr.message.to_wire()))
            for topic, wire in publish:
                self._publish(topic, wire)
            for i, osi in enumerate(osis):
                if results[i] is None:
                    results[i] = by_osi.get(osi)
        return _nonnull(results, osis)

    def _history_stock_at(self, symbol: str, t_ms: int) -> Optional["SelectedPrice"]:
        """The stock history's price at or before t_ms (thread-safe) — the cold path's first stop, so
        a contract whose underlying is already ticking skips the MDS round-trip."""
        with self._lock:
            return self.stock.at(symbol, t_ms)

    # ── helpers ─────────────────────────────────────────────────────────────────────────────────

    def _decode(self, payload: str) -> Optional[Tick]:
        now_ms = int(self._now().timestamp() * 1000)  # bounds a future timestamp only; never used as T
        tick = decode_tick(payload, now_ms)
        if tick is None:
            self._stats.bad_ticks += 1
        return tick


def _nonnull(results: list[Optional[OptionGreeksMessage]], osis: list[str]) -> list[OptionGreeksMessage]:
    # The cold path returns one message per requested osi, so a None here is a logic error.
    out = []
    for osi, r in zip(osis, results):
        if r is None:
            raise RuntimeError(f"get_greeks produced no result for {osi}")
        out.append(r)
    return out


def _tick_ts(tick: Tick) -> int:
    # A decoded tick carries exactly one of quote/trade; T is that side's own timestamp.
    if tick.quote is not None:
        return tick.quote.ts_ms
    assert tick.trade is not None
    return tick.trade.ts_ms


def _time_ts(payload: str) -> Optional[int]:
    """The ts of a cal.options.time tick: {"type":"time","ts":<epoch ms>}. None if malformed."""
    try:
        d = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(d, dict) or d.get("type") != "time":
        return None
    ts = d.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, int) or ts <= 0:
        return None
    return ts


class _Stats:
    def __init__(self) -> None:
        self._last_log = time.monotonic()
        self._reset()

    def _reset(self) -> None:
        self.passes = 0
        self.priced = 0
        self.success = 0
        self.max_s = 0.0
        self.total_s = 0.0
        self.reprices = 0
        self.bad_ticks = 0

    def record(self, n: int, ok: int, dur_s: float) -> None:
        self.passes += 1
        self.priced += n
        self.success += ok
        self.total_s += dur_s
        self.max_s = max(self.max_s, dur_s)

    def maybe_log(self, mono: float) -> None:
        if mono - self._last_log < STATS_INTERVAL_S:
            return
        if self.passes or self.reprices:
            mean_ms = 1000 * self.total_s / self.passes if self.passes else 0.0
            log.info(
                "passes=%d priced=%d success=%d pass_ms mean=%.1f max=%.1f reprices=%d bad_ticks=%d",
                self.passes, self.priced, self.success, mean_ms, 1000 * self.max_s,
                self.reprices, self.bad_ticks,
            )
        self._reset()
        self._last_log = mono
