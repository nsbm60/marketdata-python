"""Frozen domain types and enums for the option_archive package.

Mirrors ``greeks/domain.py``: no silent defaults, enums for every status/source,
frozen dataclasses. Reuses ``OptionRight`` from the greeks package rather than
redefining it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType

# Reused, not redefined — the archive speaks the same option vocabulary as greeks.
from greeks.domain import OptionRight
from greeks.occ import parse_occ

# An OSI option symbol, distinct from an arbitrary str. The only way to make one
# is to_osi(), which validates — so a contract that reaches the queue is a
# provably well-formed symbol, not a hopeful string.
OsiSymbol = NewType("OsiSymbol", str)


def to_osi(symbol: str) -> OsiSymbol:
    """Validate a standard OSI symbol at the entry point; raise on a garbled one.

    Fast-fail: a bad contract identifier never enters the queue. (A non-standard
    *deliverable* — shares_per_contract != 100 — is a separate concern handled by
    the enumeration filter with a reason code, not by this format check.)
    """
    parse_occ(symbol)  # raises ValueError if not a standard OSI
    return OsiSymbol(symbol)


__all__ = [
    "OptionRight",
    "OsiSymbol",
    "to_osi",
    "Era",
    "BandSpec",
]


class Era(str, Enum):
    """Capture era for a (contract, date), which selects the band.

    Perishable = within a year of the assumed roll-off edge at queue-build time
    (err wide, it can never be re-pulled). Routine = re-pullable if regretted.
    """

    PERISHABLE = "perishable"
    ROUTINE = "routine"


@dataclass(frozen=True)
class BandSpec:
    """Per-era capture band: moneyness half-width and DTE ceiling.

    ``moneyness_band`` is the half-width around 1.0 (0.30 = strikes within
    ±30% of spot). Both are required and validated — a zero or negative band is a
    config error, not a silently-empty capture.
    """

    moneyness_band: float
    max_dte_days: int

    def __post_init__(self) -> None:
        if not (0.0 < self.moneyness_band <= 1.0):
            raise ValueError(
                f"moneyness_band must be in (0, 1], got {self.moneyness_band!r}"
            )
        if self.max_dte_days <= 0:
            raise ValueError(f"max_dte_days must be > 0, got {self.max_dte_days!r}")
