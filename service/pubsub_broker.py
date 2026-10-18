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

import logging
import os
import socket
from dataclasses import dataclass
from typing import Optional

import zmq

from service.advertiser import Advertiser
from discovery.service_locator import ServiceLocator

log = logging.getLogger(__name__)

# ── conventions (two of these are flagged design forks; see module docstring) ──
# FLAGGED: discovery service name/key for the bus.
PUBSUB_SERVICE_NAME = ServiceLocator.PUBSUB_BUS
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


class PubSubBroker:
    """XSUB↔XPUB proxy. Bind once, then ``run()`` (blocks). ``stop()`` to unwind.

    Uses ``zmq.proxy_steerable`` with an inproc control socket so the blocking
    proxy can be told to terminate — needed for clean shutdown and for the
    in-process standalone test. A plain ``zmq.proxy`` cannot be interrupted.
    """

    def __init__(
        self,
        endpoints: BrokerEndpoints,
        *,
        xpub_sndhwm: int = XPUB_SEND_HWM,
        ctx: Optional[zmq.Context] = None,
    ):
        self._endpoints = endpoints
        self._xpub_sndhwm = xpub_sndhwm
        self._own_ctx = ctx is None
        self._ctx = ctx or zmq.Context(io_threads=1)
        self._xsub: Optional[zmq.Socket] = None
        self._xpub: Optional[zmq.Socket] = None
        self._control: Optional[zmq.Socket] = None
        self._control_endpoint = f"inproc://pubsub-broker-control-{id(self)}"

    def bind(self) -> None:
        self._xsub = self._ctx.socket(zmq.XSUB)
        self._xpub = self._ctx.socket(zmq.XPUB)
        # HWM must be set before bind to apply to subsequent connections.
        self._xpub.setsockopt(zmq.SNDHWM, self._xpub_sndhwm)
        for ep in self._endpoints.xsub:
            self._xsub.bind(ep)
        for ep in self._endpoints.xpub:
            self._xpub.bind(ep)
        # Control channel for proxy_steerable TERMINATE.
        self._control = self._ctx.socket(zmq.PAIR)
        self._control.bind(self._control_endpoint)

    def run(self) -> None:
        """Forward frames until TERMINATE. Blocks the calling thread."""
        if self._xsub is None or self._xpub is None or self._control is None:
            raise RuntimeError("call bind() before run()")
        try:
            zmq.proxy_steerable(self._xsub, self._xpub, None, self._control)
        except zmq.ContextTerminated:
            pass  # context torn down under us — an acceptable way to stop

    def stop(self) -> None:
        """Signal the running proxy to terminate (safe from another thread)."""
        killer = self._ctx.socket(zmq.PAIR)
        try:
            killer.connect(self._control_endpoint)
            killer.send(b"TERMINATE")
        finally:
            killer.close(linger=0)

    def close(self) -> None:
        for sock in (self._xsub, self._xpub, self._control):
            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
        self._xsub = self._xpub = self._control = None
        if self._own_ctx:
            try:
                self._ctx.term()
            except Exception:
                pass


def advertise_host() -> str:
    """LAN IP a remote peer would reach drogon on. Override via env."""
    env = os.environ.get("DISCOVERY_ADVERTISE_HOST")
    if env:
        return env
    # UDP connect sends nothing; it just picks the outbound interface.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.2.1", 9))
        return str(s.getsockname()[0])
    except Exception:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


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
        pub_sub=f"tcp://{host}:{XPUB_TCP_PORT}",   # pubSub = XPUB (honest subscribe endpoint)
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
    try:
        broker.run()  # blocks in zmq.proxy_steerable until TERMINATE / signal
    finally:
        advertiser.stop()
        broker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
