"""OCC/OSI parse."""

from __future__ import annotations

import pytest

from greeks.domain import OptionRight
from greeks.occ import parse_occ, strip_massive_prefix, to_massive_ticker


def test_parse_nvda() -> None:
    c = parse_occ("NVDA260417P00180000")
    assert c.root == "NVDA"
    assert c.expiry.isoformat() == "2026-04-17"
    assert c.right is OptionRight.PUT
    assert c.strike == 180.0


def test_parse_with_massive_prefix() -> None:
    c = parse_occ("O:NVDA260527C00100000")
    assert c.symbol == "NVDA260527C00100000"
    assert c.right is OptionRight.CALL
    assert c.strike == 100.0


def test_to_massive_ticker() -> None:
    assert to_massive_ticker("NVDA260527C00100000") == "O:NVDA260527C00100000"
    assert strip_massive_prefix("O:ABC") == "ABC"


def test_bad_symbol() -> None:
    with pytest.raises(ValueError):
        parse_occ("NOT_AN_OPTION")
