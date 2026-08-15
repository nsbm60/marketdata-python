"""
UDP discovery advertiser.

Broadcasts JSON service announcement to the discovery port every N seconds.
Matches Scala's ServiceAdvertiser wire format exactly.
"""

import json
import logging
import socket
import threading
import time

from service.constants import (
    DISCOVERY_BROADCAST_ADDRESS,
    DISCOVERY_PORT,
    DISCOVERY_INTERVAL_SEC,
    DISC_SERVICE_FIELD,
    DISC_HOST_FIELD,
    DISC_PUBSUB_FIELD,
    DISC_ROUTER_FIELD,
)

log = logging.getLogger(__name__)


class Advertiser:
    """Broadcasts UDP discovery announcements."""

    def __init__(
        self,
        service_name: str,
        host: str,
        pub_port: int,
        router_port: int,
        interval_sec: int = DISCOVERY_INTERVAL_SEC,
        broadcast_address: str = DISCOVERY_BROADCAST_ADDRESS,
        broadcast_port: int = DISCOVERY_PORT,
    ):
        self._service_name = service_name
        self._host = host
        self._pub_port = pub_port
        self._router_port = router_port
        self._interval_sec = interval_sec
        self._broadcast_address = broadcast_address
        self._broadcast_port = broadcast_port
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None

        self._payload = json.dumps({
            DISC_SERVICE_FIELD: service_name,
            DISC_HOST_FIELD: host,
            DISC_PUBSUB_FIELD: f"tcp://{host}:{pub_port}",
            DISC_ROUTER_FIELD: f"tcp://{host}:{router_port}",
        }).encode("utf-8")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"{self._service_name}-discovery")
        self._thread.start()

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread:
            self._thread.join(timeout=self._interval_sec + 1)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(1.0)

        consecutive_failures = 0

        try:
            while not self._shutdown.is_set():
                try:
                    sock.sendto(self._payload, (self._broadcast_address, self._broadcast_port))
                    if consecutive_failures > 0:
                        log.info("[Advertiser] Broadcast resumed after %d failures", consecutive_failures)
                        consecutive_failures = 0
                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures == 1 or consecutive_failures % 10 == 0:
                        log.warning("[Advertiser] Broadcast failed (%d consecutive): %s", consecutive_failures, e)

                self._shutdown.wait(timeout=self._interval_sec)
        finally:
            sock.close()
