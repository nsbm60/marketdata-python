"""
service_locator.py

ZMQ-based service discovery client for the MarketData system.

Subscribes to the discovery pub/sub bus and waits for service announcements.
Compatible with the Scala DiscoveryRouter and *Broadcaster classes.

Usage:
    from discovery.service_locator import ServiceLocator, ServiceEndpoint

    # Wait for ClickHouse (blocks until discovered)
    endpoint = ServiceLocator.wait_for_service("clickhouse")
    print(f"ClickHouse at {endpoint.host}:{endpoint.port}")

    # With timeout
    endpoint = ServiceLocator.wait_for_service("clickhouse", timeout_sec=30)

    # Non-blocking check
    endpoint = ServiceLocator.find_service("clickhouse")  # Returns None if not found
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import zmq

log = logging.getLogger(__name__)

# Default discovery pub/sub port (matches SystemConfig.current.discovery.pubSubPort)
DEFAULT_DISCOVERY_PORT = 6005

# Topic prefix for service discovery (matches TopicBuilder.DomainService)
SERVICE_TOPIC_PREFIX = "service."


def _default_discovery_url() -> str:
    """Get discovery URL from env or default to localhost."""
    import os
    host = os.environ.get("DISCOVERY_HOST", "localhost")
    port = os.environ.get("DISCOVERY_PORT", str(DEFAULT_DISCOVERY_PORT))
    return f"tcp://{host}:{port}"


@dataclass(frozen=True)
class ServiceEndpoint:
    """Discovered service endpoint."""
    service: str
    host: str
    port: int


class DiscoverySubscriber:
    """
    Subscribes to the ZMQ discovery bus and invokes a callback for matching services.

    The discovery bus publishes JSON messages like:
        {"service": "clickhouse", "host": "192.168.1.100", "port": 8123}

    Messages are prefixed with the service name for ZMQ topic filtering.
    """

    def __init__(
        self,
        service_name: str,
        discovery_url: str = None,
        on_discovered: Callable[[ServiceEndpoint], None] = None,
    ):
        self.service_name = service_name
        self.discovery_url = discovery_url or _default_discovery_url()
        self.on_discovered = on_discovered
        # Topic format: "service.{servicename}" (from TopicBuilder.forService)
        self._topic = f"{SERVICE_TOPIC_PREFIX}{service_name}"

        self._context: zmq.Context = None
        self._socket: zmq.Socket = None
        self._running = False
        self._thread: threading.Thread = None

    def start(self) -> None:
        """Start the subscriber in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the subscriber."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._cleanup()

    def _run(self) -> None:
        """Background thread loop."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(self.discovery_url)
        # Subscribe to messages with topic "service.{servicename}"
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self._topic)
        self._socket.setsockopt(zmq.RCVTIMEO, 500)  # 500ms timeout for graceful shutdown

        log.info(f"Discovery subscriber connected to {self.discovery_url}, topic='{self._topic}'")

        while self._running:
            try:
                msg = self._socket.recv_string()
                self._handle_message(msg)
            except zmq.Again:
                # Timeout - just loop and check _running
                continue
            except Exception as e:
                if self._running:
                    log.warning(f"Discovery subscriber error: {e}")
                    time.sleep(1)

        self._cleanup()

    def _handle_message(self, msg: str) -> None:
        """Parse and dispatch a discovery message."""
        try:
            # Message may be prefixed with topic (service name)
            # Format: "servicename {...}" or just "{...}"
            if msg.startswith("{"):
                json_str = msg
            else:
                # Strip topic prefix
                idx = msg.find("{")
                if idx < 0:
                    return
                json_str = msg[idx:]

            data = json.loads(json_str)
            service = data.get("service", "")
            host = data.get("host", "")
            port = data.get("port", 0)

            if service != self.service_name:
                return

            if not host or not port:
                log.warning(f"Malformed discovery message (missing host/port): {msg}")
                return

            endpoint = ServiceEndpoint(service=service, host=host, port=port)

            if self.on_discovered:
                self.on_discovered(endpoint)

        except json.JSONDecodeError as e:
            log.warning(f"Invalid JSON in discovery message: {e}")
        except Exception as e:
            log.warning(f"Error processing discovery message: {e}")

    def _cleanup(self) -> None:
        """Clean up ZMQ resources."""
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


class ServiceLocator:
    """
    Static utility for discovering services on the MarketData discovery bus.
    """

    # Service name constants (matches Scala ServiceNames)
    CLICKHOUSE = "clickhouse"
    MARKET_DATA = "marketData"
    IB = "ib"
    CALC = "calc"
    ALPACA_TRADING = "alpaca"
    ML_SCORER = "mlScorer"

    @staticmethod
    def wait_for_service(
        service_name: str,
        timeout_sec: float = None,
        discovery_url: str = None,
        log_interval_sec: float = 5.0,
        warn_after_sec: float = 30.0,
    ) -> ServiceEndpoint:
        """
        Wait for a service to be discovered.

        Args:
            service_name: The service to wait for (e.g., "clickhouse")
            timeout_sec: Maximum time to wait (None = wait forever)
            discovery_url: ZMQ discovery URL (default: tcp://localhost:6005)
            log_interval_sec: How often to log "still waiting" messages
            warn_after_sec: Log a warning after this many seconds

        Returns:
            ServiceEndpoint with host and port

        Raises:
            TimeoutError: If timeout_sec is set and service not found in time
        """
        result: ServiceEndpoint = None
        lock = threading.Lock()

        def on_discovered(endpoint: ServiceEndpoint):
            nonlocal result
            with lock:
                if result is None:
                    result = endpoint

        sub = DiscoverySubscriber(
            service_name=service_name,
            discovery_url=discovery_url,
            on_discovered=on_discovered,
        )
        sub.start()

        log.info(f"Waiting for {service_name} discovery...")
        start_time = time.time()
        last_log = start_time
        warned = False

        try:
            while True:
                with lock:
                    if result is not None:
                        log.info(f"Discovered {service_name}: {result.host}:{result.port}")
                        return result

                elapsed = time.time() - start_time

                if timeout_sec and elapsed > timeout_sec:
                    raise TimeoutError(f"Service '{service_name}' not discovered within {timeout_sec}s")

                now = time.time()
                if now - last_log > log_interval_sec:
                    if not warned and elapsed > warn_after_sec:
                        log.warning(
                            f"{service_name} not found after {elapsed:.0f}s - "
                            f"is the broadcaster running?"
                        )
                        warned = True
                    else:
                        log.debug(f"Still waiting for {service_name} discovery...")
                    last_log = now

                time.sleep(0.25)
        finally:
            sub.stop()

    @staticmethod
    def find_service(
        service_name: str,
        wait_sec: float = 2.0,
        discovery_url: str = None,
    ) -> ServiceEndpoint | None:
        """
        Try to find a service with a short timeout.

        Args:
            service_name: The service to find
            wait_sec: How long to wait before giving up
            discovery_url: ZMQ discovery URL

        Returns:
            ServiceEndpoint if found, None otherwise
        """
        try:
            return ServiceLocator.wait_for_service(
                service_name=service_name,
                timeout_sec=wait_sec,
                discovery_url=discovery_url,
            )
        except TimeoutError:
            return None
