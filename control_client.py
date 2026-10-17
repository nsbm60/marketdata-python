"""The control-plane request wire: DEALER -> ROUTER, one request, one reply.

Single source of truth for how a request is framed, matching ZmqBus.scala.

    request   [rid][json]
    reply     [rid][json]        (the ROUTER's identity frame is stripped
                                  by the DEALER before we see it)

Every request carries an id and every reply echoes it. A DEALER has no idea
which reply answers which request, so without the id a request that timed out
leaves its reply in the socket for the next caller to collect as its own
answer. Draining before sending does not fix that — the late reply can arrive
after the drain.

Before this module every tool framed its own request and they did not agree:
some sent a bare frame, some sent an empty delimiter first. The Scala router
papered over the difference by taking "the last non-empty frame", so nobody
found out. The layout lives here now, in one place.
"""

from __future__ import annotations

import itertools
import json
import time
import uuid
from typing import Any

import zmq

DEFAULT_TIMEOUT_MS = 5000


class ControlError(Exception):
    """The request could not be completed."""


class ControlTimeout(ControlError):
    def __init__(self, endpoint: str, timeout_ms: int):
        super().__init__(f"No reply from {endpoint} in {timeout_ms}ms")
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms


class ControlSocket:
    """A DEALER to one control endpoint. Reusable across requests.

    Use this when making several requests; use `request()` for a single one.
    Not thread-safe — a ZMQ socket belongs to one thread.
    """

    def __init__(self, endpoint: str, timeout_ms: int = DEFAULT_TIMEOUT_MS, ctx: zmq.Context | None = None):
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self._own_ctx = ctx is None
        self._ctx = ctx or zmq.Context()
        # Unique per socket, so a reply meant for a socket we have closed can
        # never match an id issued by its replacement.
        self._prefix = uuid.uuid4().hex[:8]
        self._ids = itertools.count(1)
        self._sock = self._open()

    def _open(self) -> zmq.Socket:
        s = self._ctx.socket(zmq.DEALER)
        s.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        s.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(self.endpoint)
        return s

    def request(self, payload: dict[str, Any], timeout_ms: int | None = None) -> dict[str, Any]:
        """Send one request, return the parsed reply. Raises ControlTimeout."""
        rid = f"{self._prefix}-{next(self._ids)}"
        wait_ms = self.timeout_ms if timeout_ms is None else timeout_ms

        self._sock.send_multipart([rid.encode("utf-8"), json.dumps(payload).encode("utf-8")])

        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        deadline = time.monotonic() + wait_ms / 1000.0
        while True:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                break
            if self._sock not in dict(poller.poll(remaining_ms)):
                break

            frames = self._sock.recv_multipart()
            if len(frames) < 2:
                raise ControlError(f"Malformed reply from {self.endpoint}: {len(frames)} frame(s)")

            reply_rid = frames[0].decode("utf-8")
            if reply_rid != rid:
                # A request that timed out has answered late. Not ours.
                continue

            return json.loads(frames[-1].decode("utf-8"))

        raise ControlTimeout(self.endpoint, wait_ms)

    def close(self) -> None:
        try:
            self._sock.close()
        finally:
            if self._own_ctx:
                self._ctx.term()

    def __enter__(self) -> "ControlSocket":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def request(
    endpoint: str,
    payload: dict[str, Any],
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ctx: zmq.Context | None = None,
) -> dict[str, Any]:
    """Open a socket, send one request, return the parsed reply, close."""
    with ControlSocket(endpoint, timeout_ms, ctx) as sock:
        return sock.request(payload)
