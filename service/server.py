"""
Service — first-class peer service infrastructure for Python.

A Python service using this class is indistinguishable from a Scala service:
it advertises itself the same way, responds to pings the same way, publishes
heartbeats the same way, wraps responses the same way.

Usage:
    from service import Service

    service = Service(name="optimizer", pub_port=6050, router_port=6051)
    service.register_handler("optimize", my_handler)
    service.run()
"""

import json
import logging
import signal
import threading

import zmq

from service.advertiser import Advertiser
from service.constants import HEARTBEAT_INTERVAL_SEC, SELF_PING_INTERVAL_SEC, SELF_PING_MAX_MISSED
from service.heartbeat import HeartbeatPublisher
from service.network import get_local_ipv4
from service.router import Router
from service.self_ping import SelfPingMonitor

log = logging.getLogger(__name__)


class Service:
    """First-class peer service matching Scala infrastructure patterns."""

    def __init__(
        self,
        name: str,
        pub_port: int,
        router_port: int,
        host: str | None = None,
        discovery_address: str = "255.255.255.255",
        discovery_port: int = 6001,
        discovery_interval_sec: int = 5,
        heartbeat_interval_sec: int = HEARTBEAT_INTERVAL_SEC,
        self_ping_interval_sec: int = SELF_PING_INTERVAL_SEC,
        self_ping_max_missed: int = SELF_PING_MAX_MISSED,
    ):
        self._name = name
        self._pub_port = pub_port
        self._router_port = router_port
        self._host = host or get_local_ipv4()
        self._discovery_address = discovery_address
        self._discovery_port = discovery_port
        self._discovery_interval_sec = discovery_interval_sec
        self._heartbeat_interval_sec = heartbeat_interval_sec
        self._self_ping_interval_sec = self_ping_interval_sec
        self._self_ping_max_missed = self_ping_max_missed

        self._shutdown_event = threading.Event()
        self._running = False
        self._context: zmq.Context | None = None
        self._pub_socket: zmq.Socket | None = None
        self._pub_lock = threading.Lock()

        self._router = Router(name, router_port)
        self._advertiser: Advertiser | None = None
        self._heartbeat: HeartbeatPublisher | None = None
        self._self_ping: SelfPingMonitor | None = None

    def register_handler(self, op: str, handler: callable) -> None:
        """Register a handler for an operation name."""
        self._router.register_handler(op, handler)

    def publish(self, topic: str, payload: dict) -> None:
        """Publish a message on the service's PUB socket."""
        if self._pub_socket is None:
            raise RuntimeError("Service not started — call run() first")
        data = json.dumps(payload)
        with self._pub_lock:
            self._pub_socket.send_multipart([
                topic.encode("utf-8"),
                data.encode("utf-8"),
            ])

    def run(self) -> None:
        """Start the service. Blocks until shutdown."""
        if self._running:
            raise RuntimeError(f"Service '{self._name}' is already running")
        self._running = True

        log.info("[%s] Starting service (host=%s, pub=%d, router=%d)",
                 self._name, self._host, self._pub_port, self._router_port)

        # Bind sockets
        self._context = zmq.Context()

        self._pub_socket = self._context.socket(zmq.PUB)
        self._pub_socket.setsockopt(zmq.LINGER, 0)
        self._pub_socket.bind(f"tcp://*:{self._pub_port}")
        log.info("[%s] PUB socket bound to tcp://*:%d", self._name, self._pub_port)

        self._router.bind(self._context)

        # Start background threads
        self._advertiser = Advertiser(
            service_name=self._name,
            host=self._host,
            pub_port=self._pub_port,
            router_port=self._router_port,
            interval_sec=self._discovery_interval_sec,
            broadcast_address=self._discovery_address,
            broadcast_port=self._discovery_port,
        )
        self._advertiser.start()
        log.info("[%s] Discovery advertiser started", self._name)

        self._heartbeat = HeartbeatPublisher(
            service_name=self._name,
            pub_socket=self._pub_socket,
            pub_lock=self._pub_lock,
            interval_sec=self._heartbeat_interval_sec,
        )
        self._heartbeat.start()
        log.info("[%s] Heartbeat publisher started (interval=%ds)", self._name, self._heartbeat_interval_sec)

        self._self_ping = SelfPingMonitor(
            service_name=self._name,
            router_port=self._router_port,
            on_recreate=lambda: self._router.recreate(self._context),
            interval_sec=self._self_ping_interval_sec,
            max_missed=self._self_ping_max_missed,
        )
        self._self_ping.start()
        log.info("[%s] Self-ping monitor started (interval=%ds, max_missed=%d)",
                 self._name, self._self_ping_interval_sec, self._self_ping_max_missed)

        # Wire SIGINT/SIGTERM to shutdown.
        # Only set the event — do NOT log from the signal handler.
        # logging uses locks internally; if the signal arrives mid-log-write, deadlock.
        signal.signal(signal.SIGINT, lambda *_: self._shutdown_event.set())
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown_event.set())

        self._router.freeze()
        ops = self._router.registered_ops()
        log.info("[%s] Service ready. Registered ops: %s", self._name, ops)

        # Main request loop
        while not self._shutdown_event.is_set():
            if not self._router.recv_and_dispatch():
                # No message available — brief sleep to avoid busy-spin
                self._shutdown_event.wait(timeout=0.001)

        self._cleanup()
        log.info("[%s] Service stopped", self._name)

    def shutdown(self) -> None:
        """Initiate graceful shutdown."""
        log.info("[%s] Shutdown requested", self._name)
        self._shutdown_event.set()

    def _cleanup(self) -> None:
        if self._self_ping:
            self._self_ping.stop()
        if self._heartbeat:
            self._heartbeat.stop()
        if self._advertiser:
            self._advertiser.stop()
        self._router.close()
        if self._pub_socket:
            try:
                self._pub_socket.close(linger=0)
            except Exception:
                pass
        if self._context:
            try:
                self._context.term()
            except Exception:
                pass
