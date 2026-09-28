"""Domain invariants for option_archive."""

from __future__ import annotations

import pytest

from option_archive.domain import BandSpec, Era, OptionRight, to_osi


def test_band_spec_accepts_valid() -> None:
    b = BandSpec(moneyness_band=0.30, max_dte_days=90)
    assert b.moneyness_band == 0.30
    assert b.max_dte_days == 90


@pytest.mark.parametrize("band", [0.0, -0.1, 1.5])
def test_band_spec_rejects_out_of_range_moneyness(band: float) -> None:
    with pytest.raises(ValueError):
        BandSpec(moneyness_band=band, max_dte_days=90)


@pytest.mark.parametrize("dte", [0, -1])
def test_band_spec_rejects_nonpositive_dte(dte: int) -> None:
    with pytest.raises(ValueError):
        BandSpec(moneyness_band=0.30, max_dte_days=dte)


def test_to_osi_validates() -> None:
    # a well-formed OSI round-trips as its own string value
    assert to_osi("NVDA260417C00180000") == "NVDA260417C00180000"


@pytest.mark.parametrize("bad", ["", "NVDA", "not-a-symbol", "NVDA260417X00180000"])
def test_to_osi_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError):
        to_osi(bad)


def test_enum_wire_values_are_stable() -> None:
    # These strings hit ClickHouse; pin them so a rename is a conscious act.
    assert Era.PERISHABLE.value == "perishable"
    assert Era.ROUTINE.value == "routine"
    assert OptionRight.CALL.value == "C"
