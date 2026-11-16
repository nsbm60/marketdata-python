"""Trade-basis greeks for a batch of option trades — the ONE place that turns (osi, price, spot,
trade_ts) rows into the greeks columns. Both greeks.history (the historical backfill) and the greeks
server's price_trades op call it, so neither re-implements any pricing: it parses the OSI, checks the
spot, builds the per-option inputs, makes ONE greeks.pricing.price_options call, and maps each
PricedGreeks to the greeks columns with a count of failures by reason. Nothing is defaulted.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from greeks.domain import Dividend, FailureReason, RowStatus
from greeks.occ import parse_occ
from greeks.pricing import OptionInput, PricedGreeks, price_options

log = logging.getLogger(__name__)

# The trade-basis constant recorded per row; the same value the greeks server reports as ``basis``.
BASIS_TRADE = "trade"

# The greeks columns price_trade_rows fills, exactly as option_trade stores them.
GREEKS_COLUMNS: Tuple[str, ...] = (
    "greeks_methodology", "greeks_basis", "greeks_status", "greeks_reason",
    "greeks_forward", "greeks_discount", "greeks_tte", "iv", "delta", "gamma", "theta", "vega", "rho",
    "greeks_computed_at",
)


# The per-day inputs this job needs: the SOFR→rate lookup and the dividends for an underlying. The
# server's SessionInputs satisfies this structurally (its rate_for / dividends_for), so the pricer
# stays independent of greeks_server/.
class Inputs(Protocol):
    def rate_for(self, expiry: date) -> float: ...
    def dividends_for(self, underlying: str) -> Sequence[Dividend]: ...


@dataclass(frozen=True, slots=True)
class TradeRow:
    """One trade to price: its OSI, trade price, spot at trade (None if none), and trade time."""
    osi: str
    price: float
    spot: Optional[float]
    as_of: datetime


def _as_utc(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


# The greeks columns as a single dict literal (faster than a comprehension over GREEKS_COLUMNS then
# overwrites). Both builders produce exactly the GREEKS_COLUMNS key set.
def _priced_cols(p: PricedGreeks, methodology_version: str, now: datetime) -> Dict[str, Any]:
    return {
        "greeks_methodology": methodology_version, "greeks_basis": BASIS_TRADE,
        "greeks_computed_at": now,
        "greeks_status": p.status.value,
        "greeks_reason": None if p.reason is None else p.reason.value,
        "greeks_forward": p.forward, "greeks_discount": p.discount, "greeks_tte": p.tte,
        "iv": p.iv, "delta": p.delta, "gamma": p.gamma, "theta": p.theta, "vega": p.vega, "rho": p.rho,
    }


def _failed_cols(reason: FailureReason, methodology_version: str, now: datetime) -> Dict[str, Any]:
    return {
        "greeks_methodology": methodology_version, "greeks_basis": BASIS_TRADE,
        "greeks_computed_at": now,
        "greeks_status": RowStatus.FAILURE.value, "greeks_reason": reason.value,
        "greeks_forward": None, "greeks_discount": None, "greeks_tte": None,
        "iv": None, "delta": None, "gamma": None, "theta": None, "vega": None, "rho": None,
    }


def price_trade_rows(
    trades: Sequence[TradeRow],
    inputs: Inputs,
    *,
    session_date: date,
    methodology_version: str,
    t_floor_minutes: float,
    expiry_time_et: str,
    now: datetime,
) -> Tuple[List[Dict[str, Any]], "Counter[str]"]:
    """One greeks-column dict per trade, in input order, plus a count of failures by reason. A
    non-standard OSI is NONSTANDARD_CONTRACT, an absent spot is MISSING_SPOT; the rest go through one
    price_options call. ``rate_for``/``dividends`` are for ``session_date``."""
    out: List[Optional[Dict[str, Any]]] = [None] * len(trades)
    reasons: "Counter[str]" = Counter()

    opt_inputs: List[OptionInput] = []
    staged: List[int] = []
    for i, t in enumerate(trades):
        try:
            c = parse_occ(t.osi)  # memoized
        except ValueError:
            out[i] = _failed_cols(FailureReason.NONSTANDARD_CONTRACT, methodology_version, now)
            reasons[FailureReason.NONSTANDARD_CONTRACT.value] += 1
            continue
        if t.spot is None:
            out[i] = _failed_cols(FailureReason.MISSING_SPOT, methodology_version, now)
            reasons[FailureReason.MISSING_SPOT.value] += 1
            continue
        opt_inputs.append(OptionInput(
            expiry=c.expiry, right=c.right, strike=c.strike, price=t.price, spot=t.spot,
            as_of=_as_utc(t.as_of), rate_for=inputs.rate_for, dividends=inputs.dividends_for(c.root)))
        staged.append(i)

    if opt_inputs:
        priced = price_options(opt_inputs, session_date=session_date,
                               methodology_version=methodology_version,
                               t_floor_minutes=t_floor_minutes, expiry_time_et=expiry_time_et)
        for j, i in enumerate(staged):
            p = priced[j]
            out[i] = _priced_cols(p, methodology_version, now)
            if p.status is RowStatus.FAILURE:
                assert p.reason is not None
                reasons[p.reason.value] += 1

    return [c for c in out if c is not None], reasons
