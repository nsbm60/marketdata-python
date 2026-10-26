"""Standalone proof that the pub/sub broker forwards topic-filtered frames
producer -> broker -> subscriber, over BOTH ipc and tcp.

Runs a real broker in a background thread (one instance binding ipc + tcp on
each side), then drives throwaway PUB/SUB sockets against it. No discovery, no
systemd — just the proxy. Not in the default CI testpaths; run explicitly:

    python -m pytest tests/service/test_pubsub_broker.py -q
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import zmq

from service.pubsub_broker import (
    EXIT_SELFCHECK_STALL,
    BrokerEndpoints,
    PubSubBroker,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _free_tcp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


@pytest.fixture
def broker(tmp_path):
    """A running broker bound to tmp ipc paths + ephemeral tcp ports."""
    xsub_ipc = f"ipc://{tmp_path}/xsub.ipc"
    xpub_ipc = f"ipc://{tmp_path}/xpub.ipc"
    xsub_tcp = f"tcp://127.0.0.1:{_free_tcp_port()}"
    xpub_tcp = f"tcp://127.0.0.1:{_free_tcp_port()}"
    endpoints = BrokerEndpoints(xsub=[xsub_ipc, xsub_tcp], xpub=[xpub_ipc, xpub_tcp])

    b = PubSubBroker(endpoints)
    b.bind()
    t = threading.Thread(target=b.run, name="pubsub-broker-test", daemon=True)
    t.start()

    yield {
        "endpoints": endpoints,
        "xsub": {"ipc": xsub_ipc, "tcp": xsub_tcp},
        "xpub": {"ipc": xpub_ipc, "tcp": xpub_tcp},
    }

    b.stop()
    t.join(timeout=3.0)
    b.close()


def _exchange(ctx: zmq.Context, xsub_ep: str, xpub_ep: str) -> list[bytes]:
    """PUB -> XSUB, SUB(topic 'A') <- XPUB. Return topics the SUB received.

    Sends both an 'A' and a 'B' message repeatedly (slow-joiner: the
    subscription takes time to propagate up to the publisher). Succeeds when
    the 'A' arrives; asserts 'B' never does.
    """
    pub = ctx.socket(zmq.PUB)
    pub.connect(xsub_ep)
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVTIMEO, 50)
    sub.setsockopt(zmq.SUBSCRIBE, b"A")
    sub.connect(xpub_ep)

    received: list[bytes] = []
    deadline = time.time() + 5.0
    got_a = False
    try:
        while time.time() < deadline and not got_a:
            pub.send_multipart([b"A", b"hello"])
            pub.send_multipart([b"B", b"nope"])
            time.sleep(0.05)
            while True:
                try:
                    topic, _payload = sub.recv_multipart()
                except zmq.Again:
                    break
                received.append(topic)
                if topic == b"A":
                    got_a = True
        return received
    finally:
        pub.close(linger=0)
        sub.close(linger=0)


def test_tcp_flow_topic_filtered(broker):
    ctx = zmq.Context.instance()
    received = _exchange(ctx, broker["xsub"]["tcp"], broker["xpub"]["tcp"])
    assert b"A" in received, "subscriber never received the subscribed topic over tcp"
    assert b"B" not in received, "subscriber received an unsubscribed topic over tcp"


def test_ipc_flow_topic_filtered(broker):
    ctx = zmq.Context.instance()
    received = _exchange(ctx, broker["xsub"]["ipc"], broker["xpub"]["ipc"])
    assert b"A" in received, "subscriber never received the subscribed topic over ipc"
    assert b"B" not in received, "subscriber received an unsubscribed topic over ipc"


def test_cross_transport_producer_ipc_subscriber_tcp(broker):
    """A producer on ipc and a subscriber on tcp meet at the same broker."""
    ctx = zmq.Context.instance()
    received = _exchange(ctx, broker["xsub"]["ipc"], broker["xpub"]["tcp"])
    assert b"A" in received, "ipc producer -> tcp subscriber did not flow"
    assert b"B" not in received


# ── self-check (liveness) ──


def test_selfcheck_survives_normal_operation(tmp_path):
    """While the proxy forwards normally, the self-check must NOT fire."""
    xsub_tcp = f"tcp://127.0.0.1:{_free_tcp_port()}"
    xpub_tcp = f"tcp://127.0.0.1:{_free_tcp_port()}"
    endpoints = BrokerEndpoints(xsub=[xsub_tcp], xpub=[xpub_tcp])

    stalls: list[float] = []
    b = PubSubBroker(
        endpoints,
        selfcheck_interval=0.2,
        selfcheck_timeout=1.0,
        on_stall=lambda stale_for: stalls.append(stale_for),
    )
    b.bind()
    t = threading.Thread(target=b.run, name="pubsub-broker-test", daemon=True)
    t.start()
    b.start_selfcheck()
    try:
        time.sleep(2.0)  # several intervals and well past the 1.0s timeout
        assert not stalls, f"self-check falsely fired while forwarding: {stalls}"
    finally:
        b.stop()
        t.join(timeout=3.0)
        b.close()


def test_selfcheck_exits_when_proxy_blocked(tmp_path):
    """Block the proxy and confirm the process exits non-zero.

    Reproduces the 2026-10-26 incident: the proxy stops forwarding while the
    process stays alive (here we stop the proxy loop, so no frames cross over
    TCP), and the self-check probe stops round-tripping. The real dump-and-exit
    path must take the process down with EXIT_SELFCHECK_STALL, which systemd's
    Restart=always would recover.
    """
    xsub_port = _free_tcp_port()
    xpub_port = _free_tcp_port()
    prog = textwrap.dedent(
        f"""
        import threading, time
        from service.pubsub_broker import BrokerEndpoints, PubSubBroker

        eps = BrokerEndpoints(
            xsub=["tcp://127.0.0.1:{xsub_port}"],
            xpub=["tcp://127.0.0.1:{xpub_port}"],
        )
        b = PubSubBroker(eps, selfcheck_interval=0.2, selfcheck_timeout=1.0)
        b.bind()
        t = threading.Thread(target=b.run, daemon=True)
        t.start()
        b.start_selfcheck()
        time.sleep(1.0)   # let probes round-trip so last_rx is fresh
        b.stop()          # stop the proxy: process alive, forwarding ceased
        time.sleep(30)    # self-check should os._exit long before this
        """
    )
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.run(
        [sys.executable, "-c", prog],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == EXIT_SELFCHECK_STALL, (
        f"expected exit {EXIT_SELFCHECK_STALL}, got {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # The stall log line and the faulthandler thread dump both land on stderr.
    assert "self-check" in proc.stderr, proc.stderr
    assert "pubsub-selfcheck" in proc.stderr, proc.stderr
