"""
Self-ping liveness monitor.

Creates a DEALER connected to the service's own ROUTER (via loopback).
Sends __ping__ every self_ping_interval_sec, expects __pong__ back.
After self_ping_max_missed consecutive misses, triggers socket recreation.

Matches Scala's ResilientRouter self-ping pattern.
"""

import json
import logging
import threading

import zmq

from control_client import ControlSocket, ControlTimeout
from service.constants import PING_JSON, PONG_JSON, SELF_PING_INTERVAL_SEC, SELF_PING_MAX_MISSED, SELF_PING_TIMEOUT_MS

log = logging.getLogger(__name__)


class SelfPingMonitor:
    """Monitors ROUTER liveness by pinging itself."""

    def __init__(
        self,
        service_name: str,
        router_port: int,
        on_recreate: callable,
        interval_sec: int = SELF_PING_INTERVAL_SEC,
        max_missed: int = SELF_PING_MAX_MISSED,
        timeout_ms: int = SELF_PING_TIMEOUT_MS,
    ):
        self._service_name = service_name
        self._router_port = router_port
        self._on_recreate = on_recreate
        self._interval_sec = interval_sec
        self._max_missed = max_missed
        self._timeout_ms = timeout_ms
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._ctx: zmq.Context | None = None
        self._dealer: ControlSocket | None = None

    def start(self) -> None:
        self._ctx = zmq.Context()
        self._dealer = ControlSocket(
            f"tcp://127.0.0.1:{self._router_port}", timeout_ms=self._timeout_ms, ctx=self._ctx
        )
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"{self._service_name}-selfping")
        self._thread.start()

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread:
            self._thread.join(timeout=self._interval_sec + 2)
        if self._dealer:
            try:
                self._dealer.close()
            except Exception:
                pass
        if self._ctx:
            try:
                self._ctx.term()
            except Exception:
                pass

    def _run(self) -> None:
        consecutive_missed = 0

        while not self._shutdown.is_set():
            self._shutdown.wait(timeout=self._interval_sec)
            if self._shutdown.is_set():
                break

            if self._ping():
                if consecutive_missed > 0:
                    log.info("[SelfPing] Recovered after %d missed pings", consecutive_missed)
                consecutive_missed = 0
                log.debug("[SelfPing] OK")
            else:
                consecutive_missed += 1
                log.warning("[SelfPing] Missed %d/%d", consecutive_missed, self._max_missed)

                if consecutive_missed >= self._max_missed:
                    log.error("[SelfPing] %d consecutive misses — triggering socket recreation", consecutive_missed)
                    try:
                        self._on_recreate()
                    except Exception as e:
                        log.error("[SelfPing] on_recreate() failed: %s — monitoring continues", e)
                    self._reconnect_dealer()
                    consecutive_missed = 0

    def _ping(self) -> bool:
        """Send a ping to our own ROUTER, return True if pong received.

        No drain: the pong carries the id of the ping it answers, so a stale one
        cannot be mistaken for this cycle's and mask a current failure.
        """
        try:
            return self._dealer.request(json.loads(PING_JSON)) == json.loads(PONG_JSON)
        except ControlTimeout:
            return False
        except Exception as e:
            log.debug("[SelfPing] Error: %s", e)
            return False

    def _reconnect_dealer(self) -> None:
        """Reconnect the dealer after ROUTER recreation."""
        try:
            self._dealer.close()
        except Exception:
            pass
        self._dealer = ControlSocket(
            f"tcp://127.0.0.1:{self._router_port}", timeout_ms=self._timeout_ms, ctx=self._ctx
        )
