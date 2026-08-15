"""OCC/OSI option symbol parsing (extracted for greeks; no portfolio_optimizer import)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from greeks.domain import OptionRight

# Standard OSI: ROOT(1-6 letters) + YYMMDD + C|P + strike*1000 (8 digits)
_OSI_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


@dataclass(frozen=True)
class OccContract:
    """Parsed OCC/OSI equity option symbol (no ``O:`` prefix)."""

    symbol: str
    root: str
    expiry: date
    right: OptionRight
    strike: float


def strip_massive_prefix(ticker: str) -> str:
    """``O:NVDA...`` → ``NVDA...``; bare OSI unchanged."""
    t = ticker.strip().upper()
    if t.startswith("O:"):
        return t[2:]
    return t


def to_massive_ticker(osi: str) -> str:
    """Bare OSI → Massive options ticker ``O:...``."""
    s = strip_massive_prefix(osi)
    return f"O:{s}"


def parse_occ(symbol: str) -> OccContract:
    """Parse standard OCC/OSI symbol.

    Example: ``NVDA260417P00180000`` → root NVDA, expiry 2026-04-17, put, 180.0.
    Accepts optional ``O:`` Massive prefix.
    """
    raw = strip_massive_prefix(symbol)
    m = _OSI_RE.match(raw)
    if not m:
        raise ValueError(f"not a standard OSI option symbol: {symbol!r}")
    root, yymmdd, right_s, strike_raw = m.groups()
    expiry = datetime.strptime(yymmdd, "%y%m%d").date()
    right = OptionRight.CALL if right_s == "C" else OptionRight.PUT
    strike = int(strike_raw) / 1000.0
    return OccContract(symbol=raw, root=root, expiry=expiry, right=right, strike=strike)


def is_standard_osi(symbol: str) -> bool:
    try:
        parse_occ(symbol)
        return True
    except ValueError:
        return False
