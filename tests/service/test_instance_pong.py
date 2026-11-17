"""The pong carries a per-process instance id, and self_ping accepts the new pong (matching on the
op field, not the whole reply)."""

from __future__ import annotations

import json
import uuid
from typing import Any

from control_client import ControlTimeout
from service.constants import INSTANCE_ID, OP_FIELD, OP_PONG, PING_JSON
from service.router import Router
from service.self_ping import SelfPingMonitor


def test_pong_carries_a_stable_instance_id_across_pings() -> None:
    router = Router("test", 6999)
    try:
        first = json.loads(router._handle(PING_JSON))
        second = json.loads(router._handle(PING_JSON))
    finally:
        router.close()
    assert first[OP_FIELD] == OP_PONG
    assert first["instance"] == INSTANCE_ID          # this process's id
    assert first["instance"] == second["instance"]   # stable across pings within one process
    # A full UUID, not a truncated prefix.
    assert uuid.UUID(first["instance"]).version is not None
    assert len(first["instance"]) == 36


class _FakeDealer:
    """Stands in for self_ping's ControlSocket; records how many requests it was sent."""

    def __init__(self, reply: Any) -> None:
        self._reply = reply
        self.calls = 0

    def request(self, _obj: Any) -> Any:
        self.calls += 1
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


def _monitor(reply: Any) -> tuple[SelfPingMonitor, _FakeDealer]:
    m = SelfPingMonitor("test", 6999, on_recreate=lambda: None)
    dealer = _FakeDealer(reply)
    m._dealer = dealer  # type: ignore[assignment]
    return m, dealer


def test_self_ping_accepts_the_instance_carrying_pong() -> None:
    m, dealer = _monitor({OP_FIELD: OP_PONG, "instance": str(uuid.uuid4())})
    assert m._ping() is True
    assert dealer.calls == 1


def test_self_ping_accepts_a_legacy_pong_without_instance() -> None:
    m, _ = _monitor({OP_FIELD: OP_PONG})
    assert m._ping() is True


def test_self_ping_rejects_a_non_pong_and_a_timeout() -> None:
    m_other, _ = _monitor({"ok": True})
    assert m_other._ping() is False
    m_timeout, _ = _monitor(ControlTimeout("tcp://127.0.0.1:6999", 5000))
    assert m_timeout._ping() is False
