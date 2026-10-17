"""Cross-language golden contract for option topic PRODUCERS.

The Scala `com.nsbm.marketdata.topic.TopicBuilder` is authoritative. It freezes the
exact wire strings it emits for a fixed list of representative contracts into
`topic_golden.json` (committed byte-identically here and in the Scala repo's
`src/test/resources/topic_golden.json`). This test asserts the Python
`TopicBuilder` reproduces every one byte-for-byte. The frozen file IS the contract
that keeps the two languages in sync.

If the format changes on purpose, regenerate the fixture from the Scala builder
(`sbt "Test/runMain com.nsbm.marketdata.topic.OptionTopicGoldenSpec"`) and copy it
to both repos. Any accidental drift fails this test (or the Scala spec).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from greeks.topic_builder import TopicBuilder

_FIXTURE = Path(__file__).with_name("topic_golden.json")
_GOLDEN = json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _ids() -> list[str]:
    return [f"{r['underlying']}-{r['expiry']}-{r['side']}-{r['strike']}" for r in _GOLDEN]


@pytest.mark.parametrize("row", _GOLDEN, ids=_ids())
def test_python_builder_reproduces_golden(row: dict[str, str]) -> None:
    underlying = row["underlying"]
    expiry = date.fromisoformat(row["expiry"])
    side = row["side"]
    strike = row["strike"]  # kept as a string so Decimal(str) parses it exactly

    assert TopicBuilder.for_option_quote(underlying, expiry, side, strike) == row["quote"]
    assert TopicBuilder.for_option_trade(underlying, expiry, side, strike) == row["trade"]
    assert TopicBuilder.for_option_greeks(underlying, expiry, side, strike) == row["greeks"]


def test_fixture_is_nonempty() -> None:
    # Guards against a silently-empty fixture making the parametrized test vacuous.
    assert len(_GOLDEN) >= 14


def test_equity_symbol_normalization_trims_and_uppercases() -> None:
    assert TopicBuilder.for_equity_trade(" nvda ") == "md.equity.trade.NVDA"
    assert TopicBuilder.for_equity_quote("aapl") == "md.equity.quote.AAPL"
