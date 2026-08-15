"""
ROUTER socket handler.

Binds a ZMQ ROUTER, receives multipart messages, dispatches to registered handlers.
Intercepts __ping__ at the transport layer. Wraps responses in {ok, data}/{ok, error}.

Matches Scala's ResilientRouter + CalcControlService dispatch pattern.
"""

import json
import logging
import threading

import zmq

from service.constants import (
    OP_FIELD,
    OP_PING,
    OK_FIELD,
    DATA_FIELD,
    ERROR_FIELD,
    PONG_JSON,
)

log = logging.getLogger(__name__)

# Truncate request payloads in error messages to this length
_MAX_PAYLOAD_IN_ERROR = 500


class Router:
    """ROUTER socket with handler dispatch and ping interception."""

    def __init__(self, service_name: str, router_port: int):
        self._service_name = service_name
        self._router_port = router_port
        self._handlers: dict[str, callable] = {}
        self._frozen = False  # set True by freeze(); rejects further registrations
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._lock = threading.Lock()
        self._generation = 0  # incremented on recreate; detects stale replies

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def bind(self, context: zmq.Context) -> None:
        """Bind the ROUTER socket. Called once at startup."""
        self._context = context
        with self._lock:
            self._socket = context.socket(zmq.ROUTER)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.bind(f"tcp://*:{self._router_port}")
        log.info("[Router] Bound to tcp://*:%d", self._router_port)

    def recreate(self, context: zmq.Context) -> None:
        """Close and rebind the ROUTER socket (called by self-ping on failure)."""
        log.warning("[Router] Recreating ROUTER socket")
        with self._lock:
            if self._socket:
                try:
                    self._socket.close(linger=0)
                except Exception:
                    pass
            self._socket = context.socket(zmq.ROUTER)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.bind(f"tcp://*:{self._router_port}")
            self._generation += 1
        log.info("[Router] ROUTER socket recreated on tcp://*:%d (gen=%d)", self._router_port, self._generation)

    def register_handler(self, op: str, handler: callable) -> None:
        if self._frozen:
            raise RuntimeError(f"Cannot register handler '{op}' after run() — register all handlers before calling run()")
        self._handlers[op] = handler

    def freeze(self) -> None:
        """Freeze the handler registry. Called by Service.run() before entering the request loop."""
        self._frozen = True

    def registered_ops(self) -> list[str]:
        return sorted(self._handlers.keys())

    def recv_and_dispatch(self) -> bool:
        """Receive one message, dispatch, and reply.

        Lock held for recv (brief, NOBLOCK) and send (brief). Released during
        handler execution so self-ping and recreate aren't blocked by long handlers.
        If the socket was recreated during the handler, the reply is dropped (the
        identity belongs to the old socket).
        """
        # --- Recv under lock (non-blocking, instant) ---
        with self._lock:
            try:
                frames = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return False
            except Exception as e:
                log.warning("[Router] Recv error: %s", e)
                return False

            if len(frames) < 2:
                return False

            identity = frames[0]
            payload_frame = frames[-1]
            recv_gen = self._generation

        # --- Handle outside lock (handler may take seconds) ---
        try:
            payload_str = payload_frame.decode("utf-8")
            response = self._handle(payload_str)
        except Exception as e:
            log.error("[Router] Unhandled error: %s", e)
            response = json.dumps({OK_FIELD: False, ERROR_FIELD: f"{type(e).__name__}: {e}"})

        # --- Send under lock (brief) ---
        with self._lock:
            if self._generation != recv_gen:
                log.warning("[Router] Socket recreated during handler — dropping reply (gen %d → %d)", recv_gen, self._generation)
                return True
            try:
                self._socket.send_multipart([identity, b"", response.encode("utf-8")])
            except Exception as e:
                log.warning("[Router] Reply send failed: %s", e)

        return True

    def _handle(self, payload_str: str) -> str:
        """Process a request, return JSON response string."""
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError as e:
            return json.dumps({OK_FIELD: False, ERROR_FIELD: f"JSON parse error: {e}"})

        op = payload.get(OP_FIELD, "")
        if op == OP_PING:
            return PONG_JSON

        if not op or not op.strip():
            truncated = payload_str[:_MAX_PAYLOAD_IN_ERROR]
            return json.dumps({
                OK_FIELD: False,
                ERROR_FIELD: f"empty or missing '{OP_FIELD}' field. Request: {truncated}",
            })

        handler = self._handlers.get(op)
        if handler is None:
            return json.dumps({
                OK_FIELD: False,
                ERROR_FIELD: f"unsupported op: '{op}'. Valid ops: {self.registered_ops()}",
            })

        try:
            result = handler(payload)
            return json.dumps({OK_FIELD: True, DATA_FIELD: result})
        except Exception as e:
            log.error("[Router] Handler '%s' raised: %s", op, e, exc_info=True)
            return json.dumps({OK_FIELD: False, ERROR_FIELD: f"{type(e).__name__}: {e}"})

    def close(self) -> None:
        with self._lock:
            if self._socket:
                try:
                    self._socket.close(linger=0)
                except Exception:
                    pass
                self._socket = None
