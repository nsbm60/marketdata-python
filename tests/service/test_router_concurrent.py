"""Router concurrent dispatch (opt-in): a slow op occupies one worker; pings and other ops are
still answered. Driven with a fake socket — no real ZMQ bind, no network."""

from __future__ import annotations

import json
import threading
import time
from collections import deque

import pytest
import zmq

from service.constants import OP_PING
from service.router import Router


class FakeSock:
    """A stand-in ROUTER socket: recv_multipart drains a queue (zmq.Again when empty), send records."""

    def __init__(self) -> None:
        self.incoming: deque[list[bytes]] = deque()
        self.sent: list[list[bytes]] = []

    def recv_multipart(self, flags: int = 0) -> list[bytes]:
        if self.incoming:
            return self.incoming.popleft()
        raise zmq.Again()

    def send_multipart(self, frames: list[bytes]) -> None:
        self.sent.append(frames)

    def close(self, linger: int = 0) -> None:
        pass


def _req(rid: bytes, op: str, **extra) -> list[bytes]:
    return [b"id-" + rid, rid, json.dumps({"op": op, **extra}).encode("utf-8")]


def _sent_rids(sock: FakeSock) -> set[bytes]:
    return {frames[1] for frames in sock.sent}


def _pump_until(router: Router, sock: FakeSock, predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        router.recv_and_dispatch()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"condition not met within {timeout}s; sent={_sent_rids(sock)}")


def test_slow_op_does_not_block_ping_or_other_ops() -> None:
    router = Router("test", 6999, request_workers=2)
    sock = FakeSock()
    router._socket = sock  # inject; no real bind

    started = threading.Event()
    release = threading.Event()

    def slow(_payload):
        started.set()
        assert release.wait(3.0)
        return {"slow": True}

    router.register_handler("slow", lambda p: slow(p))
    router.register_handler("fast", lambda p: {"fast": True})

    # Submit the slow op; it occupies one worker.
    sock.incoming.append(_req(b"r-slow", "slow"))
    assert router.recv_and_dispatch() is True
    assert started.wait(1.0)

    # While it blocks, a ping and a fast op must still be answered.
    sock.incoming.append(_req(b"r-ping", OP_PING))
    sock.incoming.append(_req(b"r-fast", "fast"))
    _pump_until(router, sock, lambda: {b"r-ping", b"r-fast"}.issubset(_sent_rids(sock)))
    assert b"r-slow" not in _sent_rids(sock)  # the slow op has not replied yet

    # Release it; its reply follows.
    release.set()
    _pump_until(router, sock, lambda: b"r-slow" in _sent_rids(sock))
    router.close()


def test_ping_answered_inline_with_pong() -> None:
    router = Router("test", 6999, request_workers=2)
    sock = FakeSock()
    router._socket = sock
    sock.incoming.append(_req(b"r1", OP_PING))
    assert router.recv_and_dispatch() is True
    assert _sent_rids(sock) == {b"r1"}
    assert json.loads(sock.sent[0][2].decode())["op"] == "__pong__"
    router.close()


def test_single_threaded_mode_unchanged() -> None:
    router = Router("test", 6999)  # request_workers=0 (default)
    sock = FakeSock()
    router._socket = sock
    router.register_handler("fast", lambda p: {"fast": True})
    sock.incoming.append(_req(b"r1", "fast"))
    assert router.recv_and_dispatch() is True          # recv → handle → send, all inline
    assert _sent_rids(sock) == {b"r1"}
    assert router.recv_and_dispatch() is False          # nothing left
