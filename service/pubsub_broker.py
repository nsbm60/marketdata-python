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
import hashlib
import logging
import os
import socket
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import zmq
from zmq.utils.monitor import recv_monitor_message

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

# ── instrumentation (added 2026-10-30 to diagnose the recurring silent stall) ──
# All observational and in background threads; none of it is on the proxy's forwarding path except
# the drop-safe capture socket (a PUB that drops at its HWM, so it can never block the proxy).
STATS_INTERVAL_SEC = 60.0            # the periodic stats line
SELFCHECK_PASS_LOG_SEC = 300.0       # log a self-check *pass* summary this often (5 min)
CAPTURE_SNDHWM = 200_000             # drop-safe: capture copies are dropped here, never blocked
# MDS publishes its heartbeat here (Scala serviceName "mds" -> "mds.heartbeat"). Silence raises
# an attention_item; it does NOT restart anything.
MDS_HEARTBEAT_TOPIC = b"mds.heartbeat"
HEARTBEAT_SILENCE_SEC = 60.0

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

        # ── instrumentation state ──
        self._capture: Optional[zmq.Socket] = None       # proxy sends a copy of every frame here
        self._capture_rx: Optional[zmq.Socket] = None    # drains the capture stream to count it
        self._capture_endpoint = f"inproc://pubsub-capture-{id(self)}"
        self._hb_sub: Optional[zmq.Socket] = None         # watches mds.heartbeat for silence
        self._mon_xsub: Optional[zmq.Socket] = None       # ZMQ socket-monitor streams
        self._mon_xpub: Optional[zmq.Socket] = None
        self._instr_stop = threading.Event()
        self._instr_threads: list[threading.Thread] = []
        self._stats_lock = threading.Lock()
        self._msg_counts: dict[str, int] = defaultdict(int)   # per topic-prefix, reset each stats tick
        self._peers_xsub: set[int] = set()                    # connected-peer fds, XSUB (producers in)
        self._peers_xpub: set[int] = set()                    # connected-peer fds, XPUB (subscribers out)
        self._last_heartbeat = time.monotonic()
        self._bus_stall_open = False                          # one attention_item per silence episode
        self._ch = None                                       # lazy ClickHouse client for bus_stall
        self._ch_unavailable = False

    def bind(self) -> None:
        self._xsub = self._ctx.socket(zmq.XSUB)
        self._xpub = self._ctx.socket(zmq.XPUB)
        # Socket monitors (peer connect/disconnect), attached before any connection so none is
        # missed. recv_monitor_message gives the local endpoint + fd; libzmq does not expose the
        # remote peer IP on these events, so we log endpoint+fd and count peers by fd.
        self._mon_xsub = self._xsub.get_monitor_socket()
        self._mon_xpub = self._xpub.get_monitor_socket()
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
        # Capture socket for proxy_steerable: the proxy sends a COPY of every forwarded frame here
        # (both legs). A PUB that drops at its HWM, so a slow reader can never stall the proxy — it
        # only undercounts. The reader SUB drains it to count throughput per topic prefix.
        self._capture = self._ctx.socket(zmq.PUB)
        self._capture.setsockopt(zmq.SNDHWM, CAPTURE_SNDHWM)
        self._capture.bind(self._capture_endpoint)
        self._capture_rx = self._ctx.socket(zmq.SUB)
        self._capture_rx.setsockopt(zmq.SUBSCRIBE, b"")
        self._capture_rx.connect(self._capture_endpoint)

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
        next_pass_log = now + SELFCHECK_PASS_LOG_SEC
        seq = 0
        send_times: dict[int, float] = {}   # seq -> send monotonic time, for round-trip timing
        roundtrips = 0                       # matched round-trips in the current pass-log window
        last_rtt_ms = 0.0
        poll_ms = max(1, int(min(self._selfcheck_interval, 1.0) * 1000))
        while not self._selfcheck_stop.is_set():
            now = time.monotonic()
            if now >= next_send:
                seq += 1
                send_times[seq] = now
                try:
                    self._probe_pub.send_multipart(
                        [SELFCHECK_TOPIC, str(seq).encode()]
                    )
                except zmq.ZMQError:
                    pass  # context going down under us; loop exit handles it
                next_send = now + self._selfcheck_interval
                # Drop send times older than the stall timeout (an unmatched probe = a dropped one).
                cutoff = now - self._selfcheck_timeout
                for k in [k for k, t in send_times.items() if t < cutoff]:
                    send_times.pop(k, None)
            socks = dict(poller.poll(timeout=poll_ms))
            if socks.get(self._probe_sub) == zmq.POLLIN:
                try:
                    while True:
                        parts = self._probe_sub.recv_multipart(zmq.NOBLOCK)
                        rx = time.monotonic()
                        last_rx = rx
                        if len(parts) >= 2:
                            try:
                                rseq = int(parts[1])
                            except ValueError:
                                rseq = None
                            if rseq is not None:
                                sent = send_times.pop(rseq, None)
                                if sent is not None:
                                    last_rtt_ms = (rx - sent) * 1000.0
                                    roundtrips += 1
                except zmq.Again:
                    pass
                except zmq.ZMQError:
                    break
            # Log a PASS summary every 5 min, not only on failure, so a healthy bus is visible.
            if time.monotonic() >= next_pass_log:
                log.info(
                    "pubsub broker self-check ok: %d round-trips in last %.0fs, last RTT %.1fms",
                    roundtrips, SELFCHECK_PASS_LOG_SEC, last_rtt_ms,
                )
                roundtrips = 0
                next_pass_log = time.monotonic() + SELFCHECK_PASS_LOG_SEC
            stale_for = time.monotonic() - last_rx
            if stale_for > self._selfcheck_timeout:
                self._on_stall(stale_for)
                return

    # ── instrumentation (observational; none of it restarts anything) ──

    # Monitor event code -> name, built once (zmq exposes the codes as module constants).
    _EVENT_NAMES = {
        getattr(zmq, n): n for n in dir(zmq) if n.startswith("EVENT_")
    }
    _CONNECT_EVENTS = {zmq.EVENT_ACCEPTED, zmq.EVENT_CONNECTED}
    _DISCONNECT_EVENTS = {zmq.EVENT_DISCONNECTED, zmq.EVENT_CLOSED, zmq.EVENT_ACCEPT_FAILED}

    def start_instrumentation(self) -> None:
        """Launch the observational threads (peer monitor, capture counter, stats, heartbeat watch).

        Call after ``bind()``. All daemon threads; none touches the forwarding path.
        """
        if self._xpub is None:
            raise RuntimeError("call bind() before start_instrumentation()")
        # Heartbeat watcher: a SUB on XPUB (TCP, the path clients use) subscribed to mds.heartbeat,
        # so the broker is itself a subscriber and the topic is always forwarded while the bus is
        # healthy — silence then means a real stall (or MDS down), not merely "no other subscriber".
        xpub_ep = _probe_connect_endpoint(self._endpoints.xpub)
        self._hb_sub = self._ctx.socket(zmq.SUB)
        self._hb_sub.setsockopt(zmq.SUBSCRIBE, MDS_HEARTBEAT_TOPIC)
        self._hb_sub.connect(xpub_ep)
        self._last_heartbeat = time.monotonic()

        for target, name in (
            (self._monitor_loop, "pubsub-monitor"),
            (self._capture_loop, "pubsub-capture"),
            (self._stats_loop, "pubsub-stats"),
            (self._heartbeat_loop, "pubsub-heartbeat"),
        ):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._instr_threads.append(t)
        log.info("pubsub broker instrumentation started | heartbeat-watch=%s topic=%s silence=%.0fs | "
                 "stats every %.0fs | self-check pass log every %.0fs",
                 xpub_ep, MDS_HEARTBEAT_TOPIC.decode(), HEARTBEAT_SILENCE_SEC,
                 STATS_INTERVAL_SEC, SELFCHECK_PASS_LOG_SEC)

    def _monitor_loop(self) -> None:
        """Log every peer connect/disconnect on both sides and keep per-side connected-peer counts.
        libzmq's monitor reports the local endpoint + fd, not the remote IP, so that is what we log."""
        poller = zmq.Poller()
        poller.register(self._mon_xsub, zmq.POLLIN)
        poller.register(self._mon_xpub, zmq.POLLIN)
        sides = {self._mon_xsub: ("xsub/producers", self._peers_xsub),
                 self._mon_xpub: ("xpub/subscribers", self._peers_xpub)}
        while not self._instr_stop.is_set():
            try:
                socks = dict(poller.poll(timeout=1000))
            except zmq.ZMQError:
                break
            for sock, (side, peers) in sides.items():
                if socks.get(sock) != zmq.POLLIN:
                    continue
                try:
                    evt = recv_monitor_message(sock, zmq.NOBLOCK)
                except (zmq.Again, zmq.ZMQError):
                    continue
                event = evt.get("event")
                fd = evt.get("value")
                endpoint = evt.get("endpoint", b"")
                endpoint = endpoint.decode() if isinstance(endpoint, bytes) else str(endpoint)
                name = self._EVENT_NAMES.get(event, str(event))
                if event in self._CONNECT_EVENTS:
                    with self._stats_lock:
                        peers.add(fd)
                        n = len(peers)
                    log.info("pubsub broker peer connected | %s | %s fd=%s | peers=%d", side, endpoint, fd, n)
                elif event in self._DISCONNECT_EVENTS:
                    with self._stats_lock:
                        peers.discard(fd)
                        n = len(peers)
                    log.info("pubsub broker peer disconnected | %s | %s fd=%s (%s) | peers=%d",
                             side, endpoint, fd, name, n)

    def _capture_loop(self) -> None:
        """Drain the capture stream and tally messages per topic prefix (first dot-segment). The
        capture carries both forwarded legs, so this is throughput, not a split of in vs out."""
        poller = zmq.Poller()
        poller.register(self._capture_rx, zmq.POLLIN)
        while not self._instr_stop.is_set():
            try:
                socks = dict(poller.poll(timeout=1000))
            except zmq.ZMQError:
                break
            if socks.get(self._capture_rx) != zmq.POLLIN:
                continue
            try:
                while True:
                    parts = self._capture_rx.recv_multipart(zmq.NOBLOCK)
                    if not parts:
                        continue
                    topic = parts[0].decode("utf-8", "replace")
                    prefix = topic.split(".", 1)[0] if topic else "(empty)"
                    with self._stats_lock:
                        self._msg_counts[prefix] += 1
            except zmq.Again:
                pass
            except zmq.ZMQError:
                break

    def _stats_loop(self) -> None:
        """Emit one stats line every STATS_INTERVAL_SEC: per-prefix message throughput over the
        interval, connected peers each side, and the process open-handle count."""
        while not self._instr_stop.wait(STATS_INTERVAL_SEC):
            with self._stats_lock:
                counts = dict(self._msg_counts)
                self._msg_counts.clear()
                n_xsub = len(self._peers_xsub)
                n_xpub = len(self._peers_xpub)
            by_prefix = " ".join(f"{p}={c}" for p, c in sorted(counts.items(), key=lambda kv: -kv[1])) or "(none)"
            log.info(
                "pubsub broker stats | msgs/%.0fs by prefix: %s | peers xsub=%d xpub=%d | open_handles=%s",
                STATS_INTERVAL_SEC, by_prefix, n_xsub, n_xpub, self._open_fd_count(),
            )

    def _heartbeat_loop(self) -> None:
        """Watch mds.heartbeat; when silent for HEARTBEAT_SILENCE_SEC raise a bus_stall
        attention_item and keep running (no restart). One item per silence episode."""
        poller = zmq.Poller()
        poller.register(self._hb_sub, zmq.POLLIN)
        while not self._instr_stop.is_set():
            try:
                socks = dict(poller.poll(timeout=1000))
            except zmq.ZMQError:
                break
            if socks.get(self._hb_sub) == zmq.POLLIN:
                try:
                    while True:
                        self._hb_sub.recv_multipart(zmq.NOBLOCK)
                        self._last_heartbeat = time.monotonic()
                        if self._bus_stall_open:
                            log.info("pubsub broker: mds.heartbeat resumed")
                            self._bus_stall_open = False
                except zmq.Again:
                    pass
                except zmq.ZMQError:
                    break
            silent_for = time.monotonic() - self._last_heartbeat
            if silent_for > HEARTBEAT_SILENCE_SEC and not self._bus_stall_open:
                self._bus_stall_open = True
                log.error("pubsub broker: mds.heartbeat silent for %.0fs — raising bus_stall "
                          "attention_item (not restarting)", silent_for)
                self._raise_bus_stall(silent_for)

    @staticmethod
    def _open_fd_count() -> str:
        """Process open-handle count. Linux via /proc/self/fd (drogon); 'n/a' elsewhere."""
        try:
            return str(len(os.listdir("/proc/self/fd")))
        except OSError:
            return "n/a"

    def _raise_bus_stall(self, silent_for: float) -> None:
        """Best-effort: write a bus_stall row to trading.attention_item (same shape/dedup as the
        Scala AttentionItemRepository: item_id = sha1('bus_stall|'+subject), never reopen a resolved
        one). Non-fatal — a ClickHouse problem must never take the broker down."""
        try:
            if self._ch is None:
                if self._ch_unavailable:
                    return
                from ml.shared.clickhouse import get_ch_client  # lazy: needs CLICKHOUSE_PASSWORD
                self._ch = get_ch_client()
            # Hourly subject so a fresh stall re-alerts while one episode does not spam.
            hour = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
            subject = f"mds.heartbeat|{hour}"
            item_id = hashlib.sha1(f"bus_stall|{subject}".encode("utf-8")).hexdigest()
            self._ch.command(
                "CREATE TABLE IF NOT EXISTS trading.attention_item "
                "(item_id String, kind LowCardinality(String), symbol Nullable(String), detail String, "
                " raised_at DateTime64(3,'UTC'), status Enum8('open'=1,'resolved'=2), updated_at DateTime64(3,'UTC')) "
                "ENGINE = ReplacingMergeTree(updated_at) ORDER BY (item_id)")
            resolved = self._ch.query(
                f"SELECT count() FROM trading.attention_item FINAL "
                f"WHERE item_id='{item_id}' AND status='resolved'").result_rows[0][0]
            if resolved:
                return  # never reopen a resolved item
            detail = (f"mds.heartbeat silent on the pub/sub bus for {silent_for:.0f}s "
                      f"(every subscriber would be starved); broker left running, not restarted")
            self._ch.command(
                "INSERT INTO trading.attention_item "
                "(item_id,kind,symbol,detail,raised_at,status,updated_at) "
                f"VALUES ('{item_id}','bus_stall',NULL,'{detail.replace(chr(39), chr(39)*2)}', "
                f"now64(3),'open',now64(3))")
            log.info("pubsub broker: raised bus_stall attention_item (%s)", subject)
        except Exception as exc:  # noqa: BLE001 — instrumentation must never crash the broker
            self._ch = None
            self._ch_unavailable = True
            log.warning("pubsub broker: could not raise bus_stall attention_item (%s); "
                        "is CLICKHOUSE_PASSWORD set for this unit?", exc)

    def run(self) -> None:
        """Forward frames until TERMINATE. Blocks the calling thread."""
        if self._xsub is None or self._xpub is None or self._control is None:
            raise RuntimeError("call bind() before run()")
        try:
            # capture = self._capture: the proxy sends a copy of every forwarded frame to it, which
            # the instrumentation drains to count throughput. Drop-safe PUB, so it never blocks here.
            zmq.proxy_steerable(self._xsub, self._xpub, self._capture, self._control)
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
        self._instr_stop.set()
        if self._selfcheck_thread is not None:
            self._selfcheck_thread.join(timeout=2.0)
            self._selfcheck_thread = None
        for t in self._instr_threads:
            t.join(timeout=2.0)
        self._instr_threads = []
        with self._control_lock:
            for sock in (self._xsub, self._xpub, self._control, self._control_tx,
                         self._probe_pub, self._probe_sub,
                         self._capture, self._capture_rx, self._hb_sub,
                         self._mon_xsub, self._mon_xpub):
                if sock is not None:
                    try:
                        sock.close(linger=0)
                    except Exception as exc:
                        log.warning("pubsub broker close: socket close failed (%s)", exc)
            self._xsub = self._xpub = self._control = self._control_tx = None
            self._probe_pub = self._probe_sub = None
            self._capture = self._capture_rx = self._hb_sub = None
            self._mon_xsub = self._mon_xpub = None
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
    broker.start_instrumentation()
    try:
        broker.run()  # blocks in zmq.proxy_steerable until TERMINATE / signal
    finally:
        advertiser.stop()
        broker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
