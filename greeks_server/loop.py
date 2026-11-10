"""The compute loop: one thread, one subscriber, driven by live data (greeks-server-spec.md §3-§6).

The live bus carries live ticks only, so pairing needs no session or clock. Each message does one
thing:

  - an option quote or trade  -> price that contract as of the tick's own timestamp, with the
    held live stock for its underlying. If no stock is held, hand it to the resolver to fetch the
    stock from MDS off this thread (a blocking RPC must not stall repricing) — never a failure for
    a gap in our own state;
  - an equity quote or trade  -> update the held live stock for that underlying, nothing else
    (off-hours the stock moves while option quotes stand still; in session the 5-second pass brings
    quiet strikes up to date, not every stock tick);
  - a cal.options.time tick (ts, session_open_ts) -> one full pass over the book as of ts. A new
    session_open_ts clears the book and held stock first: the open is a restart, so nothing from a
    prior session is paired with today's prices.

The book and held stock are shared with get_greeks (run on a control worker), so a lock guards them.

A failure to load the rate/dividends for a date is fatal: on_fatal stops the service and systemd
restarts it. Per-contract solver errors are isolated (compute_or_fail) and published as failures.
"""

from __future__ import annotations

import json
import logging
import queue
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
from greeks_server.held_stock import HeldStock
from greeks_server.message import OptionGreeksMessage
from greeks_server.price import SelectedPrice, quote_mid
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
        self.stock = HeldStock()
        self._session_open: Optional[int] = None  # the session_open_ts the book/held-stock belong to
        self._stats = _Stats()
        self._time_topic = TopicBuilder.for_options_time()
        # Option ticks whose underlying has no held stock: the stock is fetched from MDS off the
        # compute thread (a blocking RPC must not stall repricing) by this resolver, which coalesces
        # a burst (a chain's lease bootstrap) into one batched get_equity_prices_at.
        self._cold_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._resolver: Optional[threading.Thread] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_event.set()
        self._cold_queue.put(None)  # wake the resolver so it can exit
        sub = self._sub
        if sub is not None:
            sub.stop()

    def run(self) -> None:
        try:
            while not self._ready.wait(0.25):
                if self._stop_event.is_set():
                    return
            self._resolver = threading.Thread(target=self._resolve_loop, name="greeks-coldresolve", daemon=True)
            self._resolver.start()
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
                parsed = _time_tick(payload)
                if parsed is not None:
                    ts, session_open = parsed
                    self._on_time(ts, session_open)
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
        if tick is None or tick.quote is None:
            return  # only quotes price greeks now — there is no trade fallback
        osi, t_ms = tick.symbol, tick.quote.ts_ms
        as_of = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
        inputs = self._inputs.for_date(as_of.astimezone(ET).date())  # outside the lock; raises -> fatal
        with self._lock:
            contract = self.book.ensure(osi)
            if contract is None:
                return
            if not self.book.on_option_quote(osi, tick.quote):
                return  # an older or duplicate quote: the held price and its result stand
            opt_px = self._current_option_px(osi)
            spot_px = self.stock.price(contract.underlying)
            if opt_px is not None and spot_px is None:
                # Current quote but no live stock held for this underlying (a cold start, or
                # off-hours). Don't publish a failure for a gap in our own state — hand it to the
                # resolver to fetch the stock from MDS, off this thread. The held result stands.
                self._cold_queue.put(osi)
                return
            msg = compute_or_fail(contract, opt_px, spot_px, inputs, as_of, self._m)
            accepted = self.book.accept(msg)
            topic = greeks_topic(self._prefix, contract)
        if accepted:
            self._publish(topic, msg.to_wire())
            self._stats.reprices += 1

    def _current_option_px(self, osi: str) -> Optional[SelectedPrice]:
        """The option price for greeks: the held two-sided quote's mid, but only if the quote is
        current — timestamped at or after the session open. Held under the lock. No trade fallback;
        a stale or one-sided quote yields None (a no_current_quote failure downstream)."""
        px = quote_mid(self.book.option_quotes.get(osi))
        if px is None:
            return None
        if self._session_open is not None and px.ts_ms < self._session_open:
            return None
        return px

    def _on_equity(self, payload: str) -> None:
        tick = self._decode(payload)
        if tick is None:
            return
        with self._lock:
            if tick.quote is not None:
                self.stock.on_quote(tick.symbol, tick.quote)
            if tick.trade is not None:
                self.stock.on_trade(tick.symbol, tick.trade)

    def _on_time(self, ts_ms: int, session_open: Optional[int]) -> None:
        as_of = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        inputs = self._inputs.for_date(as_of.astimezone(ET).date())  # outside the lock; raises -> fatal
        t0 = time.perf_counter()
        published: list[tuple[str, dict[str, Any]]] = []
        ok = 0
        with self._lock:
            # The open is a restart: a new session_open_ts clears the book and held stock, so nothing
            # from the previous session is priced against today's prices. The first tick after
            # startup only records the open (what accumulated this session is kept) — a change from
            # an already-recorded open is what clears.
            if session_open is not None and session_open != self._session_open:
                if self._session_open is not None:
                    log.info("session open changed (%s -> %s): clearing %d contract(s)",
                             self._session_open, session_open, len(self.book.contracts))
                    self.book.clear()
                    self.stock.clear()
                self._session_open = session_open
            self._drop_expired(ts_ms)
            contracts = list(self.book.contracts.values())
            repriced = 0
            for c in contracts:
                opt_px = self._current_option_px(c.osi)
                if opt_px is None:
                    continue  # reprice only contracts whose held quote is current
                repriced += 1
                spot_px = self.stock.price(c.underlying)
                msg = compute_or_fail(c, opt_px, spot_px, inputs, as_of, self._m)
                if self.book.accept(msg):
                    published.append((greeks_topic(self._prefix, c), msg.to_wire()))
                    ok += msg.reason is None
        for topic, wire in published:
            self._publish(topic, wire)
        self._stats.record(repriced, ok, time.perf_counter() - t0)

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
        """One message per osi, in order. A held success is returned from the book; everything else
        (a held failure, or a contract never seen) goes through the cold path — option snapshot +
        stock from MDS — is added to the book and published. Run on a control worker, so a slow
        cold path doesn't block pings or other requests (see Router concurrent dispatch)."""
        results: dict[str, OptionGreeksMessage] = {}
        misses: list[str] = []
        with self._lock:
            for osi in osis:
                held = self.book.results.get(osi)
                if held is not None and held.reason is None:
                    results[osi] = held
                else:
                    misses.append(osi)

        if misses:
            for m in self._run_cold(misses):
                results[m.osi] = m

        out = _nonnull([results.get(osi) for osi in osis], osis)
        log.info("get_greeks: requested=%d from_book=%d cold=%d%s",
                 len(osis), len(osis) - len(misses), len(misses), _by_reason(out))
        return out

    def _run_cold(self, osis: list[str]) -> list[OptionGreeksMessage]:
        """Cold path: option snapshot + stock from MDS, add to the book, publish."""
        cold = self._cold.price(osis)  # MDS I/O, no lock
        publish: list[tuple[str, dict[str, Any]]] = []
        with self._lock:
            for cr in cold:
                contract = self.book.ensure(cr.osi)
                if contract is None:
                    continue
                if cr.quote is not None:
                    self.book.on_option_quote(cr.osi, cr.quote)
                if self.book.accept(cr.message):
                    publish.append((greeks_topic(self._prefix, contract), cr.message.to_wire()))
        for topic, wire in publish:
            self._publish(topic, wire)
        return [cr.message for cr in cold]

    def _price_with_mds_stock(self, osis: list[str]) -> list[OptionGreeksMessage]:
        """The stream resolver's work: price contracts in the book from their held option price and
        the stock at T — the held live stock if one arrived since the tick, else MDS
        get_equity_prices_at (one batched call). Accepts and publishes. A failure is published only
        if MDS itself has no print."""
        if not osis:
            return []
        plans: list[tuple[Contract, Optional[SelectedPrice], int, Optional[SelectedPrice]]] = []
        wanted: set[tuple[str, int]] = set()
        with self._lock:
            for osi in osis:
                contract = self.book.contracts.get(osi)
                if contract is None:
                    continue
                opt_px = self._current_option_px(osi)
                t_ms = opt_px.ts_ms if opt_px is not None else _held_ts(self.book, osi)
                held_stock = self.stock.price(contract.underlying) if opt_px is not None else None
                plans.append((contract, opt_px, t_ms, held_stock))
                if opt_px is not None and held_stock is None:
                    wanted.add((contract.underlying, t_ms))
        stock_mds = self._cold.stock_prices(sorted(wanted)) if wanted else {}

        out: list[OptionGreeksMessage] = []
        publish: list[tuple[str, dict[str, Any]]] = []
        for contract, opt_px, t_ms, held_stock in plans:
            as_of = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
            inputs = self._inputs.for_date(as_of.astimezone(ET).date())  # off lock; raises -> propagates to caller
            with self._lock:
                spot_px = held_stock or stock_mds.get((contract.underlying, t_ms))
                msg = compute_or_fail(contract, opt_px, spot_px, inputs, as_of, self._m)
                accepted = self.book.accept(msg)
                topic = greeks_topic(self._prefix, contract)
            out.append(msg)
            if accepted:
                publish.append((topic, msg.to_wire()))
        for topic, wire in publish:
            self._publish(topic, wire)
        return out

    def _resolve_loop(self) -> None:
        """Drain the cold queue (coalescing a burst), price the batch from MDS stock, publish. One
        blocking RPC per batch, off the compute thread. A fatal input load ends the process."""
        while not self._stop_event.is_set():
            first = self._cold_queue.get()
            if first is None:
                return
            batch = {first}
            try:
                while True:
                    nxt = self._cold_queue.get_nowait()
                    if nxt is None:
                        self._cold_queue.put(None)  # re-signal stop
                        break
                    batch.add(nxt)
            except queue.Empty:
                pass
            try:
                self._price_with_mds_stock(sorted(batch))
            except BaseException as e:  # noqa: BLE001 — same posture as the compute thread
                log.exception("cold resolver failed: %s", e)
                self._on_fatal(e)
                return

    # ── helpers ─────────────────────────────────────────────────────────────────────────────────

    def _decode(self, payload: str) -> Optional[Tick]:
        now_ms = int(self._now().timestamp() * 1000)  # bounds a future timestamp only; never used as T
        tick = decode_tick(payload, now_ms)
        if tick is None:
            self._stats.bad_ticks += 1
        return tick


def _held_ts(book: Book, osi: str) -> int:
    """The newest held option quote/trade timestamp for a contract (0 if none) — the as-of to use
    when there is no usable option price."""
    stamps = [x.ts_ms for x in (book.option_quotes.get(osi), book.option_trades.get(osi)) if x is not None]
    return max(stamps) if stamps else 0


def _by_reason(messages: list[OptionGreeksMessage]) -> str:
    """A log fragment summarising failures by reason, e.g. ' failures=missing_spot:3,below_intrinsic:1'."""
    from collections import Counter
    fails = Counter(m.reason.value for m in messages if m.reason is not None)
    if not fails:
        return ""
    return " failures=" + ",".join(f"{r}:{n}" for r, n in sorted(fails.items()))


def _nonnull(results: list[Optional[OptionGreeksMessage]], osis: list[str]) -> list[OptionGreeksMessage]:
    # The cold path returns one message per requested osi, so a None here is a logic error.
    out = []
    for osi, r in zip(osis, results):
        if r is None:
            raise RuntimeError(f"get_greeks produced no result for {osi}")
        out.append(r)
    return out


def _time_tick(payload: str) -> "Optional[tuple[int, Optional[int]]]":
    """A cal.options.time tick: {"type":"time","ts":<ms>,"session_open_ts":<ms>?}. Returns
    (ts, session_open_ts) — session_open_ts is None when absent (an older MDS), which never clears
    the book. None if the tick is malformed."""
    try:
        d = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(d, dict) or d.get("type") != "time":
        return None
    ts = d.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, int) or ts <= 0:
        return None
    so = d.get("session_open_ts")
    session_open = so if (isinstance(so, int) and not isinstance(so, bool) and so > 0) else None
    return ts, session_open


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
