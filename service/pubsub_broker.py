"""Pub/sub bus broker — a standalone, stateless XSUB↔XPUB proxy for drogon.

Producers PUB into the XSUB side; subscribers SUB out of the XPUB side;
``zmq.proxy`` forwards frames and propagates subscriptions between them. The
broker parses nothing, holds no state, and has no topic logic of its own — a
restart is instant and lossless because every producer and subscriber
reconnects on its own.

Each side binds BOTH transports:
  - ``ipc://``  — drogon-local producers/subscribers (fast, no TCP stack)
  - ``tcp://``  — off-box peers (e.g. CalcServer on the Mac)
A socket can bind many endpoints; local traffic takes IPC and remote takes TCP
transparently, with no routing decision here.

The XPUB side carries a high-water mark so a slow subscriber drops stale frames
rather than growing the broker's heap — for market data, dropping under load is
the correct behaviour (ZMQ_XPUB_NODROP is left off, so the drop is silent).
libzmq exposes NO per-subscriber HWM drop count through ``zmq.proxy_steerable``:
its ``STATISTICS`` control command reports only bytes/messages *forwarded*, not
frames dropped at the HWM, so the broker cannot observe or log drops. We note
this limitation deliberately and build no workaround.

Liveness
--------
The 2026-10-26 incident: the unit was ``active (running)`` yet forwarded nothing
(no mds/greeks heartbeats on XPUB, every subscriber reconnect-looping). systemd
cannot see that — the process never died. Two defences now guard against it:

  * **Self-check thread** (``start_selfcheck``) — every ``selfcheck_interval`` it
    PUBs a probe into the broker's own XSUB and reads it back off XPUB, *over
    TCP* (the transport every failed client uses; an IPC probe could pass while
    the TCP side is stuck). If no probe round-trips for ``selfcheck_timeout`` it
    dumps all thread stacks to the journal and ``os._exit``s non-zero, letting
    systemd ``Restart=always`` recover.
  * **Dead-peer detection** — ZMTP heartbeats + TCP keepalive on both sides so a
    half-open connection (e.g. after a peer machine reboots) is dropped rather
    than left wedged.

Advertised through the existing UDP discovery path (``service.advertiser``) as one
logical service with EXPLICIT endpoint fields:
  pubSub → XPUB tcp endpoint (subscribers connect out — the honest pub/sub endpoint)
  xpub   → XPUB tcp endpoint (same, named for the bus)
  xsub   → XSUB tcp endpoint (producers connect in)
The bus has no control plane, so it advertises NO ``router`` — `router` is never
overloaded to carry XSUB. The ipc endpoints are a drogon-local convention (fixed
paths below), not carried in discovery — a remote peer cannot use ipc, and a local
peer imports the path.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import zmq

from service.advertiser import Advertiser
from discovery import PUBSUB_BUS

log = logging.getLogger(__name__)

# ── conventions (two of these are flagged design forks; see module docstring) ──
# FLAGGED: discovery service name/key for the bus.
PUBSUB_SERVICE_NAME = PUBSUB_BUS
# FLAGGED: messages buffered per subscriber on XPUB before stale frames drop.
XPUB_SEND_HWM = 100_000

# New port block (6000–6092 are already taken across the system).
XSUB_TCP_PORT = 6100  # producers connect IN
XPUB_TCP_PORT = 6101  # subscribers connect OUT

# drogon-local IPC. systemd RuntimeDirectory=md-pubsub creates /run/md-pubsub
# with the right ownership; the env override exists for tests and dev boxes.
IPC_DIR = os.environ.get("PUBSUB_IPC_DIR", "/run/md-pubsub")
XSUB_IPC_NAME = "xsub.ipc"  # producers connect IN
XPUB_IPC_NAME = "xpub.ipc"  # subscribers connect OUT

# ── self-check (liveness) ──
# Reserved topic the probe rides on. No real subscriber subscribes to it, so
# XPUB filters the probe out to every peer but the self-check's own SUB.
SELFCHECK_TOPIC = b"__pubsub_selfcheck__"
SELFCHECK_INTERVAL_SEC = 5.0
SELFCHECK_TIMEOUT_SEC = 30.0
# Distinct, non-zero so the stall path is unmistakable in the journal / tests.
# systemd Restart=always turns this into a recovery.
EXIT_SELFCHECK_STALL = 3

# ── dead-peer detection (set on both XSUB and XPUB before bind) ──
HEARTBEAT_IVL_MS = 5_000       # ZMTP ping cadence
HEARTBEAT_TIMEOUT_MS = 15_000  # drop the peer if no pong within this
# Kernel-level backstop to the ZMTP heartbeats, for paths that wedge below ZMTP.
TCP_KEEPALIVE_IDLE_SEC = 30
TCP_KEEPALIVE_INTVL_SEC = 5
TCP_KEEPALIVE_CNT = 3


@dataclass(frozen=True)
class BrokerEndpoints:
    """Bind endpoints for each side (ipc + tcp, in bind order)."""

    xsub: list[str]  # producers connect in
    xpub: list[str]  # subscribers connect out


def default_endpoints() -> BrokerEndpoints:
    return BrokerEndpoints(
        xsub=[f"ipc://{IPC_DIR}/{XSUB_IPC_NAME}", f"tcp://*:{XSUB_TCP_PORT}"],
        xpub=[f"ipc://{IPC_DIR}/{XPUB_IPC_NAME}", f"tcp://*:{XPUB_TCP_PORT}"],
    )


def _probe_connect_endpoint(binds: list[str]) -> str:
    """Pick the endpoint the self-check probe connects to — prefer TCP.

    Every client that failed on 2026-10-26 was on TCP, so the probe must
    exercise the TCP path; an IPC probe could pass while TCP is stuck. Translate
    the wildcard bind ``tcp://*:P`` to the loopback connect ``tcp://127.0.0.1:P``.
    Falls back to the first endpoint (e.g. a test with ipc-only binds).
    """
    tcp = [b for b in binds if b.startswith("tcp://")]
    chosen = tcp[0] if tcp else binds[0]
    return chosen.replace("tcp://*:", "tcp://127.0.0.1:")


def _selfcheck_stall_exit(stale_for: float) -> None:
    """Default stall action: dump every thread's stack, then exit non-zero.

    ``faulthandler.dump_traceback`` writes to stderr, which under systemd lands
    in the journal. ``os._exit`` is deliberate: the main thread is blocked in
    ``zmq.proxy_steerable`` and will not observe a normal ``sys.exit`` raised
    from this background thread, so we force the process down.
    """
    log.error(
        "pubsub broker self-check: no probe round-trip for %.1fs (TCP path "
        "stalled); dumping thread stacks and exiting %d for systemd restart",
        stale_for, EXIT_SELFCHECK_STALL,
    )
    faulthandler.dump_traceback(all_threads=True)
    sys.stderr.flush()
    os._exit(EXIT_SELFCHECK_STALL)


class PubSubBroker:
    """XSUB↔XPUB proxy. Bind once, then ``run()`` (blocks). ``stop()`` to unwind.

    Uses ``zmq.proxy_steerable`` with an inproc control socket so the blocking
    proxy can be told to terminate (``stop``) — needed for clean shutdown and
    for the self-check stall test, which stops the proxy to make forwarding
    cease while the process stays alive. A plain ``zmq.proxy`` cannot be
    interrupted. (``proxy_steerable``'s PAUSE/RESUME commands are deliberately
    unused: on the bundled libzmq 4.3.5 they misbehave — PAUSE does not pause —
    so only the reliable TERMINATE command is wired up.)
    """

    def __init__(
        self,
        endpoints: BrokerEndpoints,
        *,
        xpub_sndhwm: int = XPUB_SEND_HWM,
        selfcheck_interval: float = SELFCHECK_INTERVAL_SEC,
        selfcheck_timeout: float = SELFCHECK_TIMEOUT_SEC,
        on_stall: Optional[Callable[[float], None]] = None,
        ctx: Optional[zmq.Context] = None,
    ):
        self._endpoints = endpoints
        self._xpub_sndhwm = xpub_sndhwm
        self._selfcheck_interval = selfcheck_interval
        self._selfcheck_timeout = selfcheck_timeout
        # Injectable so an in-process test can assert detection without killing
        # the test runner; production uses the dump-and-exit default.
        self._on_stall = on_stall or _selfcheck_stall_exit
        self._own_ctx = ctx is None
        self._ctx = ctx or zmq.Context(io_threads=1)
        self._xsub: Optional[zmq.Socket] = None
        self._xpub: Optional[zmq.Socket] = None
        self._control: Optional[zmq.Socket] = None
        self._control_tx: Optional[zmq.Socket] = None
        self._control_lock = threading.Lock()
        self._control_endpoint = f"inproc://pubsub-broker-control-{id(self)}"
        self._probe_pub: Optional[zmq.Socket] = None
        self._probe_sub: Optional[zmq.Socket] = None
        self._selfcheck_stop = threading.Event()
        self._selfcheck_thread: Optional[threading.Thread] = None

    def bind(self) -> None:
        self._xsub = self._ctx.socket(zmq.XSUB)
        self._xpub = self._ctx.socket(zmq.XPUB)
        # HWM must be set before bind to apply to subsequent connections.
        self._xpub.setsockopt(zmq.SNDHWM, self._xpub_sndhwm)
        # Dead-peer detection on both sides, before bind so it applies to every
        # connection: drop half-open peers (e.g. a rebooted box) rather than
        # leaving them wedged. ZMTP heartbeats plus a TCP-keepalive backstop.
        for sock in (self._xsub, self._xpub):
            sock.setsockopt(zmq.HEARTBEAT_IVL, HEARTBEAT_IVL_MS)
            sock.setsockopt(zmq.HEARTBEAT_TIMEOUT, HEARTBEAT_TIMEOUT_MS)
            sock.setsockopt(zmq.TCP_KEEPALIVE, 1)
            sock.setsockopt(zmq.TCP_KEEPALIVE_IDLE, TCP_KEEPALIVE_IDLE_SEC)
            sock.setsockopt(zmq.TCP_KEEPALIVE_INTVL, TCP_KEEPALIVE_INTVL_SEC)
            sock.setsockopt(zmq.TCP_KEEPALIVE_CNT, TCP_KEEPALIVE_CNT)
        for ep in self._endpoints.xsub:
            self._xsub.bind(ep)
        for ep in self._endpoints.xpub:
            self._xpub.bind(ep)
        # Control channel for the proxy_steerable TERMINATE command. The sender
        # is a single persistent PAIR connected once (not a fresh connect per
        # send) so the command is delivered reliably over inproc.
        self._control = self._ctx.socket(zmq.PAIR)
        self._control.bind(self._control_endpoint)
        self._control_tx = self._ctx.socket(zmq.PAIR)
        self._control_tx.connect(self._control_endpoint)

    def start_selfcheck(self) -> None:
        """Launch the liveness probe thread. Call after ``bind()``.

        PUBs a probe into the broker's own XSUB (TCP) and reads it back off XPUB
        (TCP); a stall for ``selfcheck_timeout`` triggers ``on_stall``.
        """
        if self._xsub is None:
            raise RuntimeError("call bind() before start_selfcheck()")
        xsub_ep = _probe_connect_endpoint(self._endpoints.xsub)
        xpub_ep = _probe_connect_endpoint(self._endpoints.xpub)
        self._probe_pub = self._ctx.socket(zmq.PUB)
        self._probe_pub.connect(xsub_ep)
        self._probe_sub = self._ctx.socket(zmq.SUB)
        self._probe_sub.setsockopt(zmq.SUBSCRIBE, SELFCHECK_TOPIC)
        self._probe_sub.connect(xpub_ep)
        self._selfcheck_thread = threading.Thread(
            target=self._selfcheck_loop, name="pubsub-selfcheck", daemon=True
        )
        self._selfcheck_thread.start()
        log.info(
            "pubsub broker self-check started | probe in=%s out=%s | "
            "interval=%.1fs timeout=%.1fs",
            xsub_ep, xpub_ep, self._selfcheck_interval, self._selfcheck_timeout,
        )

    def _selfcheck_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._probe_sub, zmq.POLLIN)
        now = time.monotonic()
        last_rx = now
        next_send = now
        seq = 0
        poll_ms = max(1, int(min(self._selfcheck_interval, 1.0) * 1000))
        while not self._selfcheck_stop.is_set():
            now = time.monotonic()
            if now >= next_send:
                seq += 1
                try:
                    self._probe_pub.send_multipart(
                        [SELFCHECK_TOPIC, str(seq).encode()]
                    )
                except zmq.ZMQError:
                    pass  # context going down under us; loop exit handles it
                next_send = now + self._selfcheck_interval
            socks = dict(poller.poll(timeout=poll_ms))
            if socks.get(self._probe_sub) == zmq.POLLIN:
                try:
                    while True:
                        self._probe_sub.recv_multipart(zmq.NOBLOCK)
                        last_rx = time.monotonic()
                except zmq.Again:
                    pass
                except zmq.ZMQError:
                    break
            stale_for = time.monotonic() - last_rx
            if stale_for > self._selfcheck_timeout:
                self._on_stall(stale_for)
                return

    def run(self) -> None:
        """Forward frames until TERMINATE. Blocks the calling thread."""
        if self._xsub is None or self._xpub is None or self._control is None:
            raise RuntimeError("call bind() before run()")
        try:
            zmq.proxy_steerable(self._xsub, self._xpub, None, self._control)
        except zmq.ContextTerminated:
            pass  # context torn down under us — an acceptable way to stop

    def _send_control(self, command: bytes) -> None:
        """Send one proxy_steerable control command (safe from any thread)."""
        with self._control_lock:
            if self._control_tx is None:
                return  # already closed
            self._control_tx.send(command)

    def stop(self) -> None:
        """Signal the running proxy to terminate (safe from another thread)."""
        self._send_control(b"TERMINATE")

    def close(self) -> None:
        self._selfcheck_stop.set()
        if self._selfcheck_thread is not None:
            self._selfcheck_thread.join(timeout=2.0)
            self._selfcheck_thread = None
        with self._control_lock:
            for sock in (self._xsub, self._xpub, self._control, self._control_tx,
                         self._probe_pub, self._probe_sub):
                if sock is not None:
                    try:
                        sock.close(linger=0)
                    except Exception as exc:
                        log.warning("pubsub broker close: socket close failed (%s)", exc)
            self._xsub = self._xpub = self._control = self._control_tx = None
            self._probe_pub = self._probe_sub = None
        if self._own_ctx:
            try:
                self._ctx.term()
            except Exception as exc:
                log.warning("pubsub broker close: context term failed (%s)", exc)


def advertise_host() -> str:
    """LAN IP a remote peer would reach drogon on. Override via env.

    Refuses to advertise a loopback (127.x) address — a subscriber on another
    box cannot reach it, which is exactly the silent failure we are hardening
    against — and fails the start instead (systemd retries).
    """
    env = os.environ.get("DISCOVERY_ADVERTISE_HOST")
    if env:
        host = env
    else:
        # UDP connect sends nothing; it just picks the outbound interface.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("192.168.2.1", 9))
            host = str(s.getsockname()[0])
        except Exception as exc:
            host = socket.gethostbyname(socket.gethostname())
            log.warning(
                "advertise_host: outbound-interface probe failed (%s); "
                "falling back to hostname lookup -> %s", exc, host,
            )
        finally:
            s.close()
    if host.startswith("127."):
        raise RuntimeError(
            f"advertise_host resolved to loopback {host!r}; refusing to "
            f"advertise an endpoint no remote peer can reach. Set "
            f"DISCOVERY_ADVERTISE_HOST to drogon's LAN IP."
        )
    return host


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    os.makedirs(IPC_DIR, exist_ok=True)  # RuntimeDirectory usually does this; be safe on dev boxes

    endpoints = default_endpoints()
    broker = PubSubBroker(endpoints)
    broker.bind()

    host = advertise_host()
    advertiser = Advertiser(
        service_name=PUBSUB_SERVICE_NAME,
        host=host,
        router_port=None,                          # no control plane — no router field
        xsub=f"tcp://{host}:{XSUB_TCP_PORT}",      # producers connect in (explicit)
        xpub=f"tcp://{host}:{XPUB_TCP_PORT}",      # subscribers connect out (explicit)
    )
    advertiser.start()

    log.info(
        "pubsub broker up | XSUB(in)=%s | XPUB(out)=%s | hwm=%d | "
        "advertised '%s' host=%s xsub=tcp:%d xpub=tcp:%d (no router)",
        endpoints.xsub, endpoints.xpub, XPUB_SEND_HWM,
        PUBSUB_SERVICE_NAME, host, XSUB_TCP_PORT, XPUB_TCP_PORT,
    )
    broker.start_selfcheck()
    try:
        broker.run()  # blocks in zmq.proxy_steerable until TERMINATE / signal
    finally:
        advertiser.stop()
        broker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
