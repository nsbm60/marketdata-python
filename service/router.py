"""
ROUTER socket handler.

Binds a ZMQ ROUTER, receives multipart messages, dispatches to registered handlers.
Intercepts __ping__ at the transport layer. Wraps responses in {ok, data}/{ok, error}.

Matches Scala's ResilientRouter + CalcControlService dispatch pattern.

Dispatch is single-threaded by default: recv_and_dispatch() recv→handle→send, one request at a
time, so a slow handler blocks the next recv (and the self-ping, which then recreates the socket and
drops the reply). A service that has a legitimately slow op opts into CONCURRENT dispatch
(request_workers > 0): the main thread still owns the socket — it recvs, answers __ping__ inline, and
sends — while other ops run on a bounded worker pool and their replies are queued back to the main
thread to send. The ZMQ socket stays single-threaded (its rule); workers never touch it. A slow op
then occupies one worker, not the whole service.
"""

import json
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

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

    def __init__(self, service_name: str, router_port: int, request_workers: int = 0):
        self._service_name = service_name
        self._router_port = router_port
        self._handlers: dict[str, callable] = {}
        self._frozen = False  # set True by freeze(); rejects further registrations
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._lock = threading.Lock()
        self._generation = 0  # incremented on recreate; detects stale replies
        # Concurrent dispatch (opt-in). 0 = single-threaded (the default, unchanged behaviour).
        self._request_workers = request_workers
        self._pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=request_workers, thread_name_prefix=f"{service_name}-op")
            if request_workers > 0 else None
        )
        # Worker replies waiting for the main thread to send: (identity, rid, recv_gen, response).
        self._replies: "queue.Queue[tuple[bytes, bytes, int, str]]" = queue.Queue()

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
        """Receive one message and process it; return True if any work was done this call.

        Concurrent mode (request_workers > 0): send any ready worker replies, recv one message,
        answer __ping__ inline, and submit any other op to the worker pool. Single-threaded mode:
        recv → handle → send inline, as before.
        """
        if self._pool is not None:
            return self._recv_and_dispatch_concurrent()
        return self._recv_and_dispatch_inline()

    # ── single-threaded (default) ───────────────────────────────────────────────────────────────

    def _recv_and_dispatch_inline(self) -> bool:
        recvd = self._recv()
        if recvd is None:
            return False
        identity, rid, payload_str, recv_gen = recvd
        self._send(identity, rid, recv_gen, self._handle_safely(payload_str))
        return True

    # ── concurrent (opt-in) ─────────────────────────────────────────────────────────────────────

    def _recv_and_dispatch_concurrent(self) -> bool:
        sent = self._drain_replies()
        recvd = self._recv()
        if recvd is None:
            return sent
        identity, rid, payload_str, recv_gen = recvd
        if _is_ping(payload_str):
            # Answered on the main thread so a full worker pool can never delay a liveness ping.
            self._send(identity, rid, recv_gen, PONG_JSON)
        else:
            assert self._pool is not None
            self._pool.submit(self._run_op, identity, rid, recv_gen, payload_str)
        return True

    def _run_op(self, identity: bytes, rid: bytes, recv_gen: int, payload_str: str) -> None:
        """Worker: run the handler, queue the reply for the main thread to send. Never touches the socket."""
        self._replies.put((identity, rid, recv_gen, self._handle_safely(payload_str)))

    def _drain_replies(self) -> bool:
        """Send every ready worker reply (main thread only). True if any were sent."""
        sent = False
        while True:
            try:
                identity, rid, recv_gen, response = self._replies.get_nowait()
            except queue.Empty:
                return sent
            self._send(identity, rid, recv_gen, response)
            sent = True

    # ── socket I/O (main thread only) ───────────────────────────────────────────────────────────

    def _recv(self) -> "tuple[bytes, bytes, str, int] | None":
        """Non-blocking recv of one [identity][rid][json] request; None if nothing is waiting."""
        with self._lock:
            try:
                frames = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return None
            except Exception as e:
                log.warning("[Router] Recv error: %s", e)
                return None
            if len(frames) < 3:
                log.warning("[Router] Request with %d frame(s), expected [identity][rid][json]", len(frames))
                return None
            return frames[0], frames[1], frames[-1].decode("utf-8"), self._generation

    def _send(self, identity: bytes, rid: bytes, recv_gen: int, response: str) -> None:
        """Send one reply; drop it if the socket was recreated since the request was received."""
        with self._lock:
            if self._generation != recv_gen:
                log.warning("[Router] Socket recreated during handler — dropping reply (gen %d → %d)", recv_gen, self._generation)
                return
            try:
                self._socket.send_multipart([identity, rid, response.encode("utf-8")])
            except Exception as e:
                log.warning("[Router] Reply send failed: %s", e)

    def _handle_safely(self, payload_str: str) -> str:
        try:
            return self._handle(payload_str)
        except Exception as e:
            log.error("[Router] Unhandled error: %s", e)
            return json.dumps({OK_FIELD: False, ERROR_FIELD: f"{type(e).__name__}: {e}"})

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
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            if self._socket:
                try:
                    self._socket.close(linger=0)
                except Exception:
                    pass
                self._socket = None


def _is_ping(payload_str: str) -> bool:
    try:
        return json.loads(payload_str).get(OP_FIELD) == OP_PING
    except (json.JSONDecodeError, AttributeError):
        return False
