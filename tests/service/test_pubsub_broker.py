"""Standalone proof that the pub/sub broker forwards topic-filtered frames
producer -> broker -> subscriber, over BOTH ipc and tcp.

Runs a real broker in a background thread (one instance binding ipc + tcp on
each side), then drives throwaway PUB/SUB sockets against it. No discovery, no
systemd — just the proxy. Not in the default CI testpaths; run explicitly:

    python -m pytest tests/service/test_pubsub_broker.py -q
"""
from __future__ import annotations

import socket
import threading
import time

import pytest
import zmq

from service.pubsub_broker import BrokerEndpoints, PubSubBroker


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
