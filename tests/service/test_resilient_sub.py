"""ResilientSub (service/resilient_sub.py): data delivery, heartbeat handling, reconnect, callbacks.

Real sockets over loopback TCP, short intervals. No discovery, no broker: a bare PUB stands in for
the bus. Run with:  python -m pytest tests/service/test_resilient_sub.py -q
"""

from __future__ import annotations

import socket
import time
from typing import Callable

import pytest
import zmq

from service.resilient_sub import ResilientSub

HB = "mds.heartbeat"


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


@pytest.fixture
def pub():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    port = _free_port()
    sock.bind(f"tcp://127.0.0.1:{port}")
    yield sock, f"tcp://127.0.0.1:{port}"
    sock.close(linger=0)
    ctx.term()


def _send(sock: zmq.Socket, topic: str, payload: str = "{}") -> None:
    sock.send_multipart([topic.encode(), payload.encode()])


def _pump(sub: ResilientSub, until: Callable[[], bool], tick: Callable[[], None], timeout: float = 5.0) -> list[tuple[str, str]]:
    """Call tick() and sub.recv() until `until()` holds; return the data messages seen."""
    seen: list[tuple[str, str]] = []
    deadline = time.monotonic() + timeout
    while not until():
        assert time.monotonic() < deadline, "condition not reached"
        tick()
        topic, payload = sub.recv()
        if topic:
            seen.append((topic, payload))
    return seen


def test_data_returned_heartbeat_hidden_live_fires_once(pub) -> None:
    sock, ep = pub
    live: list[int] = []
    sub = ResilientSub("t", ep, ["md.option.quote."], HB, heartbeat_interval_s=1.0,
                       on_subscription_live=lambda: live.append(1), recv_timeout_s=0.05)
    try:
        _pump(sub, lambda: bool(live), lambda: _send(sock, HB))
        got: list[tuple[str, str]] = []
        deadline = time.monotonic() + 3
        while not got:
            assert time.monotonic() < deadline
            _send(sock, "md.equity.quote.NVDA", "{}")  # not subscribed: never delivered
            _send(sock, HB)
            _send(sock, "md.option.quote.NVDA.2026-11-20.C.180_00", '{"bid":1}')
            topic, payload = sub.recv()
            assert not topic.startswith(HB)
            assert not topic.startswith("md.equity.")
            if topic:
                got.append((topic, payload))
        assert got[0] == ("md.option.quote.NVDA.2026-11-20.C.180_00", '{"bid":1}')
        assert live == [1]
        assert sub.reconnects == 0
    finally:
        sub.close()


def test_reconnects_after_missed_heartbeats_and_live_refires(pub) -> None:
    sock, ep = pub
    live: list[int] = []
    sub = ResilientSub("t", ep, ["md."], HB, heartbeat_interval_s=0.2, max_missed=2,
                       reconnect_delay_s=0.05, on_subscription_live=lambda: live.append(1),
                       recv_timeout_s=0.05)
    try:
        _pump(sub, lambda: len(live) == 1, lambda: _send(sock, HB))
        # Silence: no heartbeats for well over 0.4 s -> reconnect.
        _pump(sub, lambda: sub.reconnects >= 1, lambda: None)
        assert len(live) == 1
        # Heartbeats again: the new connection reports live a second time.
        _pump(sub, lambda: len(live) == 2, lambda: _send(sock, HB))
    finally:
        sub.close()


def test_heartbeat_resumed_after_gap_without_reconnect(pub) -> None:
    sock, ep = pub
    live: list[int] = []
    resumed: list[int] = []
    sub = ResilientSub("t", ep, ["md."], HB, heartbeat_interval_s=0.1, max_missed=50,
                       on_subscription_live=lambda: live.append(1),
                       on_heartbeat_resumed=lambda: resumed.append(1), recv_timeout_s=0.02)
    try:
        _pump(sub, lambda: bool(live), lambda: _send(sock, HB))
        # Go quiet for 0.4 s (> 1.5 * 0.1 s, far under 50 * 0.1 s), draining any backlog.
        t_end = time.monotonic() + 0.4
        while time.monotonic() < t_end:
            sub.recv()
        resumed.clear()
        _pump(sub, lambda: bool(resumed), lambda: _send(sock, HB))
        assert sub.reconnects == 0
        assert live == [1]
    finally:
        sub.close()


def test_callback_exception_does_not_break_recv(pub) -> None:
    sock, ep = pub
    calls: list[int] = []

    def boom() -> None:
        calls.append(1)
        raise RuntimeError("boom")

    sub = ResilientSub("t", ep, ["md."], HB, heartbeat_interval_s=1.0,
                       on_subscription_live=boom, recv_timeout_s=0.05)
    try:
        _pump(sub, lambda: bool(calls), lambda: _send(sock, HB))
        got = _pump_data(sub, sock)
        assert got == ("md.x", "1")
    finally:
        sub.close()


def _pump_data(sub: ResilientSub, sock: zmq.Socket) -> tuple[str, str]:
    deadline = time.monotonic() + 3
    while True:
        assert time.monotonic() < deadline
        _send(sock, "md.x", "1")
        topic, payload = sub.recv()
        if topic:
            return topic, payload


def test_recv_timeout_bounds_a_quiet_recv(pub) -> None:
    _, ep = pub
    sub = ResilientSub("t", ep, ["md."], HB, heartbeat_interval_s=10.0, recv_timeout_s=0.05)
    try:
        t0 = time.monotonic()
        assert sub.recv() == ("", "")
        assert time.monotonic() - t0 < 1.0
    finally:
        sub.close()


def test_stop_makes_recv_return_immediately(pub) -> None:
    _, ep = pub
    sub = ResilientSub("t", ep, ["md."], HB, heartbeat_interval_s=10.0)
    sub.stop()
    t0 = time.monotonic()
    assert sub.recv() == ("", "")
    assert time.monotonic() - t0 < 0.5
    assert sub.running is False
    sub.close()


def test_bad_endpoint_neither_hangs_nor_spins() -> None:
    t0 = time.monotonic()
    sub = ResilientSub("t", "bogus-endpoint", ["md."], HB, heartbeat_interval_s=0.1, max_missed=2,
                       reconnect_delay_s=0.01, recv_timeout_s=0.05)
    assert time.monotonic() - t0 < 1.0             # constructor returns
    calls = 0
    t1 = time.monotonic()
    while time.monotonic() - t1 < 0.5:
        assert sub.recv() == ("", "")
        calls += 1
    assert calls < 20                              # waits per call, no busy loop
    assert sub.reconnects >= 1                     # and keeps retrying the connect
    sub.close()
