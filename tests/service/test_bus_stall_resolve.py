"""The pub/sub broker raises bus_stall with real (now64) times and resolves its own item when the
bus recovers. Methods are exercised directly against a fake ClickHouse client — no sockets."""

from __future__ import annotations

from typing import Any

from service.pubsub_broker import PubSubBroker


class _FakeCh:
    def __init__(self, rows: list[list[Any]]) -> None:
        self.commands: list[str] = []
        self._rows = rows

    def command(self, sql: str) -> None:
        self.commands.append(sql)

    def query(self, _sql: str) -> Any:
        rows = self._rows

        class _R:
            result_rows = rows

        return _R()


def _broker(ch: _FakeCh) -> PubSubBroker:
    b = PubSubBroker.__new__(PubSubBroker)  # bypass __init__ (no sockets); set only what the methods use
    b._ch = ch  # type: ignore[attr-defined]
    b._ch_unavailable = False  # type: ignore[attr-defined]
    return b


def test_raise_bus_stall_uses_now64_and_opens() -> None:
    ch = _FakeCh(rows=[[0]])  # the resolved-count probe returns 0 -> not resolved -> proceed to insert
    _broker(ch)._raise_bus_stall(120.0)
    inserts = [c for c in ch.commands if "INSERT INTO trading.attention_item" in c]
    assert len(inserts) == 1
    assert "'bus_stall'" in inserts[0] and "'open'" in inserts[0]
    assert inserts[0].count("now64(3)") == 2  # raised_at and updated_at both real server time


def test_resolve_bus_stall_resolves_open_items_preserving_raised_at() -> None:
    # One open bus_stall item: (item_id, symbol, detail, raised_ms)
    ch = _FakeCh(rows=[["abc123", None, "bus silent", 1_700_000_000_000]])
    _broker(ch)._resolve_bus_stall()
    inserts = [c for c in ch.commands if "INSERT INTO trading.attention_item" in c]
    assert len(inserts) == 1
    assert "'resolved'" in inserts[0]
    assert "'abc123'" in inserts[0]
    assert "fromUnixTimestamp64Milli(1700000000000)" in inserts[0]  # original raised_at preserved


def test_resolve_bus_stall_noop_when_nothing_open() -> None:
    ch = _FakeCh(rows=[])
    _broker(ch)._resolve_bus_stall()
    assert not [c for c in ch.commands if "INSERT" in c]
