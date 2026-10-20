"""ResilientSub — SUB socket with heartbeat-based liveness. Port of Scala zmq/ResilientSub.scala.

Every subscriber on the pub/sub bus is a ResilientSub (bus-subscriber-liveness-spec.md). A
subscriber that only waits on recv() can't tell a quiet topic from a dead socket; this one watches
the heartbeat of the service that publishes its topics and reconnects when it stops.

Behaviour matches Scala:
  - recv() returns (topic, payload) for data, and ("", "") on a heartbeat or a receive timeout.
    Heartbeats are infrastructure and are never returned to the caller.
  - On a receive timeout or a receive error, if no heartbeat has arrived for
    heartbeat_interval_s * max_missed, close the socket, wait reconnect_delay_s and connect again
    exactly as at start (same code path).
  - on_subscription_live fires on the first heartbeat of each connection, reconnects included.
    Connecting is not subscribing: the first heartbeat is the first proof messages arrive.
  - on_heartbeat_resumed fires when heartbeats come back after a gap of more than 1.5 intervals
    without a reconnect.
  - A callback that raises is logged; it never breaks recv().

One addition: recv_timeout_s (default: the heartbeat interval, as in Scala). A caller that must
wake on its own clock even when the bus is quiet passes a shorter value. The heartbeat check still
runs on every timeout.

Not ported: Scala's dynamic subscribeTopic/unsubscribeTopic. No Python subscriber needs them.

Threading: one thread owns the socket and calls recv() and close(). stop() is the one call that is
safe from another thread: it makes the next recv() return at once (and cuts a reconnect wait short),
after which the owner closes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional, Sequence

import zmq

log = logging.getLogger(__name__)


def _noop() -> None:
    return None


class ResilientSub:
    def __init__(
        self,
        name: str,
        endpoint: str,
        data_prefixes: Sequence[str],
        heartbeat_topic: str,
        heartbeat_interval_s: float = 10.0,
        max_missed: int = 2,
        reconnect_delay_s: float = 2.0,
        on_subscription_live: Callable[[], None] = _noop,
        on_heartbeat_resumed: Callable[[], None] = _noop,
        recv_timeout_s: Optional[float] = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not heartbeat_topic:
            raise ValueError("heartbeat_topic is required")
        if heartbeat_interval_s <= 0 or max_missed < 1:
            raise ValueError("heartbeat_interval_s must be > 0 and max_missed >= 1")
        self._name = name
        self._endpoint = endpoint
        self._data_prefixes = list(data_prefixes)
        self._heartbeat_topic = heartbeat_topic
        self._interval_s = float(heartbeat_interval_s)
        self._max_missed = int(max_missed)
        self._reconnect_delay_s = float(reconnect_delay_s)
        self._on_live = on_subscription_live
        self._on_resumed = on_heartbeat_resumed
        self._recv_timeout_s = float(recv_timeout_s) if recv_timeout_s is not None else self._interval_s
        self._clock = clock

        self._running = True
        self._stop = threading.Event()
        self._ctx: Optional[zmq.Context[Any]] = None
        self._socket: Optional[zmq.Socket[Any]] = None
        self._subscription_live = False
        self._last_heartbeat = self._clock()
        self.reconnects = 0  # diagnostics / tests

        self._connect()

    # ── public ────────────────────────────────────────────────────────────────────────────────

    def recv(self) -> tuple[str, str]:
        """Next DATA message as (topic, payload), or ("", "") on heartbeat/timeout."""
        if not self._running or self._socket is None:
            return "", ""
        try:
            if self._socket.poll(int(self._recv_timeout_s * 1000), zmq.POLLIN) == 0:
                self._check_heartbeat()
                return "", ""
            frames = self._socket.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            self._check_heartbeat()
            return "", ""
        except Exception as e:  # socket closed under us, context terminated, …
            if self._running:
                log.warning("[%s] recv error: %s", self._name, e)
                self._check_heartbeat()
            return "", ""

        if len(frames) < 2:
            log.warning("[%s] message with %d frame(s), expected [topic][payload]", self._name, len(frames))
            return "", ""
        topic = frames[0].decode("utf-8", errors="replace")
        if topic.startswith(self._heartbeat_topic):
            self._on_heartbeat()
            return "", ""
        return topic, frames[1].decode("utf-8", errors="replace")

    def stop(self) -> None:
        """Thread-safe: ask the owner's recv() loop to stop. The owner then calls close()."""
        self._running = False
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._running

    def close(self) -> None:
        """Owner thread only."""
        self.stop()
        self._close_socket()
        log.info("[%s] Closed", self._name)

    # ── internals ─────────────────────────────────────────────────────────────────────────────

    def _on_heartbeat(self) -> None:
        now = self._clock()
        since_last = now - self._last_heartbeat
        self._last_heartbeat = now
        if not self._subscription_live:
            self._subscription_live = True
            log.info("[%s] subscription live", self._name)
            self._fire(self._on_live, "on_subscription_live")
        elif since_last > self._interval_s * 1.5:
            log.info("[%s] heartbeats resumed after %.1fs gap", self._name, since_last)
            self._fire(self._on_resumed, "on_heartbeat_resumed")

    def _fire(self, cb: Callable[[], None], label: str) -> None:
        try:
            cb()
        except Exception as e:
            log.warning("[%s] %s failed: %s", self._name, label, e)

    def _check_heartbeat(self) -> None:
        silent = self._clock() - self._last_heartbeat
        threshold = self._interval_s * self._max_missed
        if silent > threshold:
            log.warning(
                "[%s] No heartbeat for %.1fs (threshold %.1fs) - reconnecting...",
                self._name, silent, threshold,
            )
            self._reconnect()

    def _connect(self) -> None:
        try:
            self._ctx = zmq.Context()
            sock = self._ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(self._endpoint)
            for prefix in [*self._data_prefixes, self._heartbeat_topic]:
                sock.setsockopt_string(zmq.SUBSCRIBE, prefix)
            self._socket = sock
        except Exception as e:
            log.error("[%s] Connection failed: %s", self._name, e)
            self._close_socket()
        # Give the publisher time to send a first heartbeat, and re-arm the live callback:
        # a reconnect has the same blind window as a first connect.
        self._last_heartbeat = self._clock()
        self._subscription_live = False
        log.info(
            "[%s] Connected to %s for: %s (heartbeat=%s), waiting for heartbeat...",
            self._name, self._endpoint, ", ".join(self._data_prefixes), self._heartbeat_topic,
        )

    def _reconnect(self) -> None:
        self._close_socket()
        if not self._running:
            return
        if self._stop.wait(self._reconnect_delay_s):
            return
        self._connect()
        self.reconnects += 1
        log.info("[%s] Reconnected to %s", self._name, self._endpoint)

    def _close_socket(self) -> None:
        sock, ctx = self._socket, self._ctx
        self._socket, self._ctx = None, None
        if sock is not None:
            try:
                sock.close(linger=0)
            except Exception:
                pass
        if ctx is not None:
            try:
                ctx.term()
            except Exception:
                pass
