"""
service_locator.py - Service discovery via ZMQ subscription.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import zmq

log = logging.getLogger(__name__)

DEFAULT_DISCOVERY_PORT = 6005
SERVICE_TOPIC_PREFIX = "service."


def _default_discovery_url() -> str:
    import os
    host = os.environ.get("DISCOVERY_HOST", "localhost")
    port = os.environ.get("DISCOVERY_PORT", str(DEFAULT_DISCOVERY_PORT))
    return f"tcp://{host}:{port}"


@dataclass(frozen=True)
class ServiceEndpoint:
    service: str
    host: str
    port: int
    pub_sub: str = ""
    router: str = ""


class DiscoverySubscriber:

    def __init__(
        self,
        service_name: str,
        discovery_url: str = None,
        on_discovered: Callable[[ServiceEndpoint], None] = None,
    ):
        self.service_name = service_name
        self.discovery_url = discovery_url or _default_discovery_url()
        self.on_discovered = on_discovered
        self._topic = f"{SERVICE_TOPIC_PREFIX}{service_name}"

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
                    topic   = frames[0].decode("utf-8")
                    payload = frames[1].decode("utf-8")
                    self._handle_message(payload, topic)

            except zmq.Again:
                continue
            except Exception as e:
                if self._running:
                    log.warning(f"[Discovery] Receive error: {e}")
                    time.sleep(1)

        self._cleanup()

    def _handle_message(self, msg: str, topic: str) -> None:
        try:
            data    = json.loads(msg)
            host    = data.get("host", "")
            pub_sub = data.get("pubSub", "")
            router  = data.get("router", "")
            port    = data.get("port", 0)

            if not port and pub_sub:
                try:
                    port = int(pub_sub.split(":")[-1])
                except (ValueError, IndexError):
                    pass

            if not host or not port:
                log.warning(f"Malformed discovery message: {msg}")
                return

            # Use service name from topic, not from payload
            # Topic format is "service.{name}" -- extract the name
            service = topic.removeprefix(SERVICE_TOPIC_PREFIX)

            endpoint = ServiceEndpoint(
                service = service,
                host    = host,
                port    = port,
                pub_sub = pub_sub,
                router  = router,
            )

            if self.on_discovered:
                self.on_discovered(endpoint)

        except json.JSONDecodeError as e:
            log.warning(f"JSON decode error: {e}")

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


class ServiceLocator:

    CLICKHOUSE     = "clickhouse"
    MARKET_DATA    = "marketdata"
    IB             = "ib"
    CALC           = "calc"
    ALPACA_TRADING = "alpaca"
    ML_SCORER      = "mlscorer"

    @staticmethod
    def wait_for_service(
        service_name: str,
        timeout_sec: float = None,
        discovery_url: str = None,
        log_interval_sec: float = 5.0,
        warn_after_sec: float = 30.0,
    ) -> ServiceEndpoint:
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
        last_log   = start_time
        warned     = False

        try:
            while True:
                with lock:
                    if result is not None:
                        log.info(f"Discovered {service_name}: "
                                 f"{result.host}:{result.port} "
                                 f"pubSub={result.pub_sub}")
                        return result

                elapsed = time.time() - start_time

                if timeout_sec and elapsed > timeout_sec:
                    raise TimeoutError(
                        f"Service '{service_name}' not discovered "
                        f"within {timeout_sec}s"
                    )

                now = time.time()
                if now - last_log > log_interval_sec:
                    if not warned and elapsed > warn_after_sec:
                        log.warning(
                            f"{service_name} not found after {elapsed:.0f}s - "
                            f"is the broadcaster running?"
                        )
                        warned = True
                    else:
                        log.debug(f"Still waiting for {service_name}...")
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
        try:
            return ServiceLocator.wait_for_service(
                service_name=service_name,
                timeout_sec=wait_sec,
                discovery_url=discovery_url,
            )
        except TimeoutError:
            return None
