"""ComputeLoop.step(): session gating, coalescing, clock tick, publish topics, fatal loads.

Driven directly through step() with a fake clock and calendar — no sockets, no threads.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import pytest

from greeks.config import load_config
from greeks.domain import RowStatus
from greeks_server.calendar import Session
from greeks_server.inputs import InputsHolder, SessionInputs
from greeks_server.loop import ComputeLoop

from gs_helpers import SESSION, sofr_flat

OPEN = datetime(2026, 10, 20, 13, 30, tzinfo=timezone.utc)
SESS = Session(SESSION, OPEN, OPEN + timedelta(hours=6, minutes=30))
OPT_TOPIC = "md.option.quote.NVDA.2026-11-20.C.180_00"


class Clock:
    def __init__(self) -> None:
        self.t = OPEN + timedelta(hours=1)
        self.mono = 1000.0

    def advance(self, s: float) -> None:
        self.t += timedelta(seconds=s)
        self.mono += s


class Harness:
    def __init__(self, in_session: bool = True, loader_fails: bool = False) -> None:
        self.cfg = load_config()
        self.cfg = replace(self.cfg, server=replace(self.cfg.server, topic_prefix="shadow."))
        self.clock = Clock()
        self.in_session = in_session
        self.published: list[tuple[str, dict[str, Any]]] = []
        self.loads: list[date] = []
        self.fatal: list[BaseException] = []

        def loader(d: date) -> SessionInputs:
            self.loads.append(d)
            if loader_fails:
                raise ValueError("no SOFR")
            return SessionInputs(session_date=d, sofr=sofr_flat(), dividends={})

        self.loop = ComputeLoop(
            self.cfg,
            make_subscriber=lambda: None,  # type: ignore[arg-type,return-value]
            publish=lambda t, p: self.published.append((t, p)),
            publisher_ready=threading.Event(),
            inputs=InputsHolder(loader),
            on_fatal=self.fatal.append,
            now=lambda: self.clock.t,
            monotonic=lambda: self.clock.mono,
            session_fn=self._session,
        )

    def _session(self, t: datetime) -> Optional[Session]:
        return SESS if self.in_session else None

    def quote(self, topic: str, symbol: str, bid: float, ask: float) -> None:
        ts = int(self.clock.t.timestamp() * 1000)
        self.loop.step(topic, json.dumps({"type": "quote", "symbol": symbol, "bid": bid, "ask": ask, "timestamp": str(ts)}))

    def idle(self) -> None:
        self.loop.step("", "")


def test_off_session_drops_ticks_and_loads_nothing() -> None:
    h = Harness(in_session=False)
    h.quote(OPT_TOPIC, "NVDA261120C00180000", 9.0, 9.2)
    assert not h.loop.book.contracts and not h.published and not h.loads


def test_open_loads_inputs_once_and_publishes_with_prefix() -> None:
    h = Harness()
    h.quote("md.equity.quote.NVDA", "NVDA", 181.0, 181.02)
    h.quote(OPT_TOPIC, "NVDA261120C00180000", 9.0, 9.2)
    assert h.loads == [SESSION]
    topics = [t for t, _ in h.published]
    assert topics == ["shadow.md.option.greeks.NVDA.2026-11-20.C.180_00"]
    msg = h.published[0][1]
    assert msg["status"] == RowStatus.SUCCESS.value and msg["price_kind"] == "mid"


def test_coalescing_one_pass_per_interval() -> None:
    h = Harness()
    h.quote("md.equity.quote.NVDA", "NVDA", 181.0, 181.02)
    h.quote(OPT_TOPIC, "NVDA261120C00180000", 9.0, 9.2)
    n = len(h.published)
    for i in range(10):  # ten underlying ticks inside 100 ms
        h.clock.advance(0.01)
        h.quote("md.equity.quote.NVDA", "NVDA", 181.0 + i * 0.01, 181.02 + i * 0.01)
    assert len(h.published) == n  # still inside min_pass_interval (250 ms)
    h.clock.advance(0.2)
    h.idle()
    assert len(h.published) == n + 1  # one pass for all of them


def test_clock_tick_reprices_everything() -> None:
    h = Harness()
    h.quote("md.equity.quote.NVDA", "NVDA", 181.0, 181.02)
    h.quote(OPT_TOPIC, "NVDA261120C00180000", 9.0, 9.2)
    h.quote("md.option.quote.NVDA.2026-11-20.P.170_00", "NVDA261120P00170000", 3.0, 3.1)
    h.clock.advance(1.0)
    h.idle()
    before = len(h.published)
    h.clock.advance(h.cfg.server.clock_tick_s)
    h.idle()
    assert len(h.published) == before + 2


def test_no_spot_publishes_failure() -> None:
    h = Harness()
    h.quote(OPT_TOPIC, "NVDA261120C00180000", 9.0, 9.2)
    assert h.published[-1][1]["reason"] == "missing_spot"


def test_close_clears_state() -> None:
    h = Harness()
    h.quote(OPT_TOPIC, "NVDA261120C00180000", 9.0, 9.2)
    h.in_session = False
    h.idle()
    assert h.loop.session is None and not h.loop.book.contracts
    n = len(h.published)
    h.clock.advance(30)
    h.idle()
    assert len(h.published) == n


def test_failed_load_is_fatal_through_run() -> None:
    h = Harness(loader_fails=True)
    with pytest.raises(ValueError):
        h.idle()  # step() raises; run() turns that into on_fatal


def test_run_reports_fatal_and_returns() -> None:
    h = Harness(loader_fails=True)

    class Sub:
        def recv(self) -> tuple[str, str]:
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
    """The loop is a real Thread: stop() then join() returns (no shadowing of Thread internals)."""
    h = Harness()
    h.loop.start()
    h.loop.stop()
    h.loop.join(timeout=2.0)
    assert not h.loop.is_alive()
