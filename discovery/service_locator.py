"""
service_locator.py - Service discovery via ZMQ subscription.

ServiceLocator is the one entry point: it knows every service and the shape it advertises, and
returns a typed result per shape (ControlEndpoint / HostPort / BusEndpoints). A non-member argument
raises TypeError — the runtime equivalent of the Scala compile error.

The transport (DiscoverySubscriber) does not parse: it hands the raw payload string to its callback,
matching the Scala DiscoverySubscriber. Field extraction and the "malformed" decision live in the
facade's per-shape parsers.
"""

import json
import logging
import threading
import time
from typing import Callable, Optional, TypeVar

import zmq

from greeks.topic_builder import TopicBuilder

from .services import (
    BusEndpoints,
    ControlEndpoint,
    ControlService,
    EndpointService,
    HostPort,
    PUBSUB_BUS,
)

log = logging.getLogger(__name__)

DEFAULT_DISCOVERY_PORT = 6005
SERVICE_TOPIC_PREFIX = TopicBuilder.service_prefix()  # "service." — sourced from the one topic home


def _default_discovery_url() -> str:
    import os
    host = os.environ.get("DISCOVERY_HOST", "localhost")
    port = os.environ.get("DISCOVERY_PORT", str(DEFAULT_DISCOVERY_PORT))
    return f"tcp://{host}:{port}"


class DiscoverySubscriber:
    """Subscribes to one service's discovery topic and hands each raw payload to a callback.

    It does not parse: the callback receives the JSON string verbatim, as in the Scala
    DiscoverySubscriber. Parsing and validation are the facade's job.
    """

    def __init__(
        self,
        service_name: str,
        discovery_url: str = None,
        on_message: Callable[[str], None] = None,
    ):
        self.service_name = service_name
        self.discovery_url = discovery_url or _default_discovery_url()
        self.on_message = on_message
        # Lowercases the name (via TopicBuilder) to match the wire topic `service.<name lower>`,
        # so a caller passing canonical "marketData" still matches "service.marketdata".
        self._topic = TopicBuilder.for_service(service_name)

        self._context: zmq.Context = None
        self._socket: zmq.Socket = None
        self._running = False
        self._thread: threading.Thread = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._cleanup()

    def _run(self) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(self.discovery_url)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self._topic)
        self._socket.setsockopt(zmq.RCVTIMEO, 500)

        log.debug(f"[Discovery] Connected to {self.discovery_url}, "
                  f"subscribed to topic='{self._topic}'")

        while self._running:
            try:
                frames = self._socket.recv_multipart()
                if len(frames) >= 2:
                    payload = frames[1].decode("utf-8")
                    if self.on_message:
                        self.on_message(payload)
            except zmq.Again:
                continue
            except Exception as e:
                if self._running:
                    log.warning(f"[Discovery] Receive error: {e}")
                    time.sleep(1)

        self._cleanup()

    def _cleanup(self) -> None:
        if self._socket:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        if self._context:
            try:
                self._context.term()
            except Exception:
                pass
            self._context = None


T = TypeVar("T")


class ServiceLocator:
    """Resolve a service to its typed endpoint. The one entry point for discovery."""

    @staticmethod
    def control(s: ControlService, timeout_sec: Optional[float] = None) -> ControlEndpoint:
        if not isinstance(s, ControlService):
            raise TypeError(f"control() takes a ControlService, got {s!r}")
        return ServiceLocator._await(s.value, _parse_control, timeout_sec)

    @staticmethod
    def endpoint(s: EndpointService, timeout_sec: Optional[float] = None) -> HostPort:
        if not isinstance(s, EndpointService):
            raise TypeError(f"endpoint() takes an EndpointService, got {s!r}")
        return ServiceLocator._await(s.value, _parse_endpoint, timeout_sec)

    @staticmethod
    def bus(timeout_sec: Optional[float] = None) -> BusEndpoints:
        return ServiceLocator._await(PUBSUB_BUS, _parse_bus, timeout_sec)

    @staticmethod
    def _await(
        wire: str,
        parse: Callable[[dict], T],
        timeout_sec: Optional[float],
        log_interval_sec: float = 5.0,
        warn_after_sec: float = 30.0,
    ) -> T:
        def _endpoint_str(r) -> str:
            host = getattr(r, "host", None)
            port = getattr(r, "port", None) or getattr(r, "router", None)
            return f"{host}:{port}" if host and port else repr(r)

        # The name check and parse run on the subscriber thread; the result (or the failure) is
        # handed back under the lock. A record for another service on the same topic prefix is
        # ignored. A malformed record for the requested service raises — a bug in the advertiser.
        state: dict = {}
        lock = threading.Lock()

        def on_message(msg: str) -> None:
            with lock:
                if "result" in state or "error" in state:
                    return
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError as e:
                    state["error"] = ValueError(f"'{wire}' announcement is not JSON: {e} — {msg}")
                    return
                # Name check first: ignore an announcement for another service on the same prefix.
                if str(data.get("service", "")).lower() != wire.lower():
                    return
                try:
                    state["result"] = parse(data)
                except ValueError as e:
                    state["error"] = ValueError(f"'{wire}' announcement malformed: {e} — {msg}")

        sub = DiscoverySubscriber(service_name=wire, on_message=on_message)
        sub.start()
        log.info(f"Waiting for '{wire}' discovery..."
                 + (f" (timeout={timeout_sec}s)" if timeout_sec else ""))

        start_time = time.time()
        last_log = start_time
        warned = False

        try:
            while True:
                with lock:
                    if "error" in state:
                        raise state["error"]
                    if "result" in state:
                        log.info(f"✅ Discovered '{wire}': {_endpoint_str(state['result'])} "
                                 f"in {time.time() - start_time:.2f}s")
                        return state["result"]

                elapsed = time.time() - start_time
                if timeout_sec and elapsed > timeout_sec:
                    raise TimeoutError(f"Service '{wire}' not discovered within {timeout_sec}s")

                now = time.time()
                if now - last_log > log_interval_sec:
                    if not warned and elapsed > warn_after_sec:
                        log.warning(f"'{wire}' not found after {elapsed:.0f}s — is the broadcaster running?")
                        warned = True
                    else:
                        log.debug(f"Still waiting for '{wire}'...")
                    last_log = now

                time.sleep(0.25)
        finally:
            sub.stop()


# ── Per-shape parsers ──────────────────────────────────────────────────────────────────────────
# Raise ValueError when a record for the requested service is malformed. Extra fields are ignored,
# so a control service still running pre-change code (which also sends a `pubSub`) parses fine.

def _parse_control(data: dict) -> ControlEndpoint:
    host = data.get("host", "")
    router = data.get("router", "")
    if not host or not router:
        raise ValueError("missing or empty host/router")
    return ControlEndpoint(host=host, router=router)


def _parse_endpoint(data: dict) -> HostPort:
    host = data.get("host", "")
    port = data.get("port", 0)
    if not host:
        raise ValueError("missing host")
    try:
        port = int(port)
    except (TypeError, ValueError):
        raise ValueError(f"port is not an int ({port!r})")
    if port <= 0:
        raise ValueError(f"port <= 0 ({port})")
    return HostPort(host=host, port=port)


def _parse_bus(data: dict) -> BusEndpoints:
    if "router" in data:
        raise ValueError("carries a 'router' — the bus has no control plane")
    xsub = data.get("xsub", "")
    xpub = data.get("xpub", "")
    if not xsub or not xpub:
        raise ValueError("missing or empty xsub/xpub")
    return BusEndpoints(xsub=xsub, xpub=xpub)
