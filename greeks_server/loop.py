"""The compute loop: one thread, one subscriber, one dirty set (greeks-server-spec.md §5, §9).

Each iteration: receive (bounded by min_pass_interval), check the calendar, then
  - entering a session: load SOFR and dividends, start an empty book;
  - leaving it: clear the book, stop publishing;
  - in session: apply the tick, mark everything dirty every clock_tick_s, and run a pass when the
    dirty set is non-empty and min_pass_interval has elapsed since the last one.

The calendar check is the only session signal, so a pre-open start, a mid-session start and a
restart all take the same path. Off-session ticks are dropped. A failure to load session inputs
is fatal: on_fatal stops the service and the process exits for systemd to restart.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol

from greeks.config import GreeksConfig
from greeks.topic_builder import TopicBuilder
from greeks_server.book import Book
from greeks_server.calendar import Session, session_at
from greeks_server.compute import Contract, Methodology, compute
from greeks_server.inputs import InputsHolder, SessionInputs
from greeks_server.price import select_price
from greeks_server.wire import decode_tick

log = logging.getLogger(__name__)

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
        on_fatal: Callable[[BaseException], None],
        *,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        monotonic: Callable[[], float] = time.monotonic,
        session_fn: Callable[[datetime], Optional[Session]] = session_at,
    ) -> None:
        super().__init__(name="greeks-compute", daemon=True)
        self._cfg = cfg
        self._m = methodology(cfg)
        self._prefix = cfg.server.topic_prefix
        self._tick_s = cfg.server.clock_tick_s
        self._pass_s = cfg.server.min_pass_interval_ms / 1000.0
        self._make_sub = make_subscriber
        self._publish = publish
        self._ready = publisher_ready
        self._inputs = inputs
        self._on_fatal = on_fatal
        self._now = now
        self._mono = monotonic
        self._session_fn = session_fn
        self._stop_event = threading.Event()
        self._sub: Optional[Subscriber] = None

        self.book = Book()
        self.session: Optional[Session] = None
        self._session_inputs: Optional[SessionInputs] = None
        self._last_tick = 0.0
        self._last_pass = 0.0
        self._stats = _Stats()

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

    # ── one iteration (public for tests) ──────────────────────────────────────────────────────

    def step(self, topic: str, payload: str) -> None:
        now = self._now()
        self._gate(now)
        if self.session is None:
            return
        if topic:
            self._apply(topic, payload)
        mono = self._mono()
        if mono - self._last_tick >= self._tick_s:
            self.book.mark_all_dirty()
            self._last_tick = mono
        if self.book.dirty and mono - self._last_pass >= self._pass_s:
            self._run_pass(now)
            self._last_pass = self._mono()
        self._stats.maybe_log(mono)

    def _gate(self, now: datetime) -> None:
        s = self._session_fn(now)
        if s is not None and (self.session is None or self.session.date != s.date):
            inputs = self._inputs.get_for(s.date)  # raises -> fatal
            self.book = Book()
            self.session = s
            self._session_inputs = inputs
            self._last_tick = self._mono()
            log.info("session open %s (opened %s): computing", s.date, s.open_utc.isoformat())
        elif s is None and self.session is not None:
            log.info("session %s closed: %s contracts this session", self.session.date, len(self.book.contracts))
            self.session = None
            self._session_inputs = None
            self.book = Book()

    def _apply(self, topic: str, payload: str) -> None:
        tick = decode_tick(payload)
        if tick is None:
            self._stats.bad_ticks += 1
            return
        if topic.startswith(TopicBuilder.prefix_option_quotes()) or topic.startswith(
            TopicBuilder.prefix_option_trades()
        ):
            self.book.on_option(tick)
        elif topic.startswith(TopicBuilder.prefix_equity_quotes()) or topic.startswith(
            TopicBuilder.prefix_equity_trades()
        ):
            self.book.on_equity(tick)

    def _run_pass(self, now: datetime) -> None:
        assert self.session is not None and self._session_inputs is not None
        open_ms = self.session.open_ms
        inputs = self._session_inputs
        b = self.book
        t0 = time.perf_counter()
        ok = 0
        contracts = b.take_dirty()
        for c in contracts:
            spot_px = select_price(b.equity_quotes.get(c.underlying), b.equity_trades.get(c.underlying), open_ms)
            opt_px = select_price(b.option_quotes.get(c.osi), b.option_trades.get(c.osi), open_ms)
            msg = compute(c, opt_px, spot_px, inputs, now, self._m)
            self._publish(greeks_topic(self._prefix, c), msg.to_wire())
            ok += msg.reason is None
        self._stats.record(len(contracts), ok, time.perf_counter() - t0)


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
        if self.passes:
            log.info(
                "passes=%d priced=%d success=%d pass_ms mean=%.1f max=%.1f bad_ticks=%d",
                self.passes, self.priced, self.success,
                1000 * self.total_s / self.passes, 1000 * self.max_s, self.bad_ticks,
            )
        self._reset()
        self._last_log = mono
