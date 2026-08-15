#!/usr/bin/env python3
"""Integration test: verify MDS option subscriptions via the `option_subscriptions` control op.

Connects to the live MDS control ROUTER (via ZMQ discovery), queries {"op": "option_subscriptions"},
and inspects what MDS is currently providing option data for — the set that drives the option poller
AND the option stream. This confirms subscriptions are actually being tracked, not merely that ticks
are flowing.

Requires MDS + discovery to be running and reachable.

Run standalone:
    python tests/integration/test_option_subscriptions.py
or under pytest:
    pytest tests/integration/test_option_subscriptions.py

To exercise it end-to-end: open an options chain in the UI, run this — that underlying's contracts
should appear with TTLs near the lease duration. Close the chain; after the grace period they lapse.
"""
import json
import re
import sys
import zmq

try:
    from discovery.service_locator import ServiceLocator
except ModuleNotFoundError:
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from discovery.service_locator import ServiceLocator

# Canonical service name; ServiceLocator (via TopicBuilder.for_service) lowercases it to match
# the wire topic `service.marketdata`.
SERVICE = "marketData"
_OSI = re.compile(r"^([A-Z]+)\d{6}[CP]\d{8}$")


def _underlying(symbol: str) -> str:
    m = _OSI.match(symbol)
    return m.group(1) if m else "?"


def query_option_subscriptions(timeout_sec: int = 15) -> dict:
    """Discover MDS, send the option_subscriptions op over a DEALER, return the parsed reply."""
    endpoint = ServiceLocator.wait_for_service(SERVICE, timeout_sec=timeout_sec)
    ctx = zmq.Context()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, 5000)
    dealer.setsockopt(zmq.LINGER, 0)
    dealer.connect(endpoint.router)
    try:
        dealer.send_multipart([b"", json.dumps({"op": "option_subscriptions"}).encode()])
        frames = dealer.recv_multipart()
    finally:
        dealer.close()
        ctx.term()
    return json.loads(frames[-1].decode("utf-8"))


def test_option_subscriptions() -> dict:
    """Assert the reply is well-formed; return the data block for reporting."""
    reply = query_option_subscriptions()
    assert reply.get("ok") is True, f"option_subscriptions not ok: {reply}"
    data = reply["data"]
    leases = data["leases"]
    assert isinstance(leases, list), f"leases is not a list: {data}"
    assert data["count"] == len(leases), f"count {data['count']} != len(leases) {len(leases)}"
    for leg in leases:
        assert {"symbol", "expires_in_ms", "missed_inquiries"} <= leg.keys(), f"malformed entry: {leg}"
    return data


def _report(data: dict) -> None:
    leases = data["leases"]
    print(f"subscribed contracts: {data['count']}")
    if not leases:
        print("  (none — open an options chain in the UI, then re-run)")
        return
    by_und: dict[str, list] = {}
    for leg in leases:
        by_und.setdefault(_underlying(leg["symbol"]), []).append(leg)
    print(f"  underlyings: {len(by_und)}")
    for und in sorted(by_und):
        legs = by_und[und]
        ttls = [leg["expires_in_ms"] for leg in legs]
        grace = sum(1 for leg in legs if leg["missed_inquiries"] > 0)
        expired = sum(1 for leg in legs if leg["expires_in_ms"] <= 0)
        flags = ""
        if grace:
            flags += f"  WARN {grace} in grace"
        if expired:
            flags += f"  WARN {expired} expired"
        print(f"    {und:<8} {len(legs):>4} contracts  TTL {min(ttls) / 1000:.0f}-{max(ttls) / 1000:.0f}s{flags}")


if __name__ == "__main__":
    try:
        data = test_option_subscriptions()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    except Exception as e:  # discovery/connection failure
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(2)
    _report(data)
    print("OK: option_subscriptions reply well-formed")
