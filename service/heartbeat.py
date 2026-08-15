"""
Heartbeat publisher.

Publishes {"ts": "<ISO-8601>"} to {service_name}.heartbeat on the PUB socket
every heartbeat_interval_sec seconds. Matches Scala's TopicPublisher pattern.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone

import zmq

from service.constants import HEARTBEAT_TOPIC_SUFFIX, HEARTBEAT_TS_FIELD, HEARTBEAT_INTERVAL_SEC

log = logging.getLogger(__name__)


class HeartbeatPublisher:
    """Publishes heartbeats on a PUB socket."""

    def __init__(
        self,
        service_name: str,
        pub_socket: zmq.Socket,
        pub_lock: threading.Lock,
        interval_sec: int = HEARTBEAT_INTERVAL_SEC,
    ):
        self._service_name = service_name
        self._pub_socket = pub_socket
        self._pub_lock = pub_lock
        self._interval_sec = interval_sec
        self._topic = f"{service_name}.{HEARTBEAT_TOPIC_SUFFIX}"
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"{self._service_name}-heartbeat")
        self._thread.start()

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread:
            self._thread.join(timeout=self._interval_sec + 1)

    def _run(self) -> None:
        # Immediate first heartbeat
        self._publish()

        while not self._shutdown.is_set():
            self._shutdown.wait(timeout=self._interval_sec)
            if not self._shutdown.is_set():
                self._publish()

    def _publish(self) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        payload = json.dumps({HEARTBEAT_TS_FIELD: ts})
        try:
            with self._pub_lock:
                self._pub_socket.send_multipart([
                    self._topic.encode("utf-8"),
                    payload.encode("utf-8"),
                ])
            log.debug("[Heartbeat] Published on %s", self._topic)
        except Exception as e:
            log.warning("[Heartbeat] Publish failed: %s", e)
