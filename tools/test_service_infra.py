#!/usr/bin/env python3
"""
Test harness for the service infrastructure library.
Covers acceptance criteria layers 1-4.

Run the echo service in one terminal:
    python tools/run_echo_service.py

Run this test in another:
    python tools/test_service_infra.py
"""

import json
import sys
import time
import zmq

# Add parent dir to path for discovery package
sys.path.insert(0, ".")
from discovery.service_locator import ServiceLocator

SERVICE_NAME = "echo_test"


def main():
    print("=" * 60)
    print("Service Infrastructure Test")
    print("=" * 60)

    # Layer 1: Discovery
    print("\n--- Layer 1: Discovery ---")
    print(f"Waiting for '{SERVICE_NAME}' service...")
    try:
        endpoint = ServiceLocator.wait_for_service(SERVICE_NAME, timeout_sec=15)
    except TimeoutError:
        print(f"FAIL: Service '{SERVICE_NAME}' not discovered within 15s")
        print("Is run_echo_service.py running?")
        sys.exit(1)

    print(f"OK: Discovered {endpoint.service} at {endpoint.host}:{endpoint.port}")
    print(f"    pubSub={endpoint.pub_sub}")
    print(f"    router={endpoint.router}")

    ctx = zmq.Context()

    # Layer 2.5: Heartbeat
    print("\n--- Layer 2.5: Heartbeat ---")
    sub = ctx.socket(zmq.SUB)
    sub.connect(endpoint.pub_sub)
    sub.setsockopt_string(zmq.SUBSCRIBE, f"{SERVICE_NAME}.heartbeat")
    sub.setsockopt(zmq.RCVTIMEO, 15000)  # 15s timeout

    try:
        frames = sub.recv_multipart()
        topic = frames[0].decode("utf-8")
        payload = json.loads(frames[1].decode("utf-8"))
        assert topic == f"{SERVICE_NAME}.heartbeat", f"Expected topic '{SERVICE_NAME}.heartbeat', got '{topic}'"
        assert "ts" in payload, f"Expected 'ts' field in heartbeat, got {payload}"
        print(f"OK: Received heartbeat on '{topic}' — ts={payload['ts']}")
    except zmq.Again:
        print("FAIL: No heartbeat received within 15s")
        sys.exit(1)
    finally:
        sub.close()

    # Layer 3: External ping
    print("\n--- Layer 3: External Ping ---")
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, 5000)
    dealer.setsockopt(zmq.LINGER, 0)
    dealer.connect(endpoint.router)

    dealer.send_multipart([b"", json.dumps({"op": "__ping__"}).encode()])
    try:
        frames = dealer.recv_multipart()
        reply = json.loads(frames[-1].decode("utf-8"))
        assert reply == {"op": "__pong__"}, f"Expected pong, got {reply}"
        print(f"OK: Ping/pong successful — {reply}")
    except zmq.Again:
        print("FAIL: No pong received within 5s")
        sys.exit(1)

    # Layer 4: Echo handler
    print("\n--- Layer 4: Echo Handler ---")

    # 4a: Success case
    request = {"op": "echo", "msg": "hello"}
    dealer.send_multipart([b"", json.dumps(request).encode()])
    frames = dealer.recv_multipart()
    reply = json.loads(frames[-1].decode("utf-8"))
    assert reply["ok"] is True, f"Expected ok=true, got {reply}"
    assert reply["data"]["received"] == request, f"Expected echo of request, got {reply['data']}"
    print(f"OK: Echo handler — received={reply['data']}")

    # 4b: Unknown op
    dealer.send_multipart([b"", json.dumps({"op": "nonexistent"}).encode()])
    frames = dealer.recv_multipart()
    reply = json.loads(frames[-1].decode("utf-8"))
    assert reply["ok"] is False, f"Expected ok=false, got {reply}"
    assert "unsupported op" in reply["error"], f"Expected 'unsupported op' in error, got {reply['error']}"
    print(f"OK: Unknown op — error={reply['error']}")

    # 4c: Missing op
    dealer.send_multipart([b"", json.dumps({"foo": "bar"}).encode()])
    frames = dealer.recv_multipart()
    reply = json.loads(frames[-1].decode("utf-8"))
    assert reply["ok"] is False, f"Expected ok=false, got {reply}"
    assert "missing" in reply["error"].lower() or "empty" in reply["error"].lower(), f"Expected missing op error, got {reply['error']}"
    print(f"OK: Missing op — error={reply['error']}")

    # 4d: Handler exception
    dealer.send_multipart([b"", json.dumps({"op": "fail"}).encode()])
    frames = dealer.recv_multipart()
    reply = json.loads(frames[-1].decode("utf-8"))
    assert reply["ok"] is False, f"Expected ok=false, got {reply}"
    assert "ValueError" in reply["error"], f"Expected ValueError in error, got {reply['error']}"
    print(f"OK: Handler exception — error={reply['error']}")

    dealer.close()
    ctx.term()

    print("\n" + "=" * 60)
    print("ALL LAYERS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
