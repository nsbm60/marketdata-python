"""Forward and discount construction (carry only).

**Approved escrowed-dividend form (do not substitute):**

.. math::

    F = S \\cdot e^{r T} - \\sum_i \\mathrm{div}_i \\cdot e^{r (T - t_i)}

where ``t_i`` is the ACT/365 year fraction from trade time to dividend ex-date
``i``, and only dividends with ex-date strictly inside the remaining life
``(0 < t_i < T)`` are included. Each dividend is future-valued from its own
ex-date, not present-valued from spot then grown.

Rejected (do not implement):

- ``F = (S - PV(divs)) * exp(r T)``
- ``F = S * exp(r T) - FV(divs)`` with a single cashflow date for all dividends

Discount:

.. math::

    D = e^{-r T}

with continuous rate ``r`` (from SOFR series via :mod:`greeks.forwards.sofr`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Sequence
from zoneinfo import ZoneInfo

from greeks.domain import Dividend
from greeks.solver.time import expiry_instant_utc, years_to_expiry

_ET = ZoneInfo("America/New_York")
_SECONDS_PER_ACT365_YEAR = 365.0 * 86400.0


@dataclass(frozen=True)
class CarryResult:
    """Spot carry outputs for one trade / inversion input."""

    spot: float
    forward: float
    discount: float
    continuous_rate: float
    time_to_expiry: float
    dividends_applied: tuple[Dividend, ...]


def discount_factor(continuous_rate: float, time_to_expiry: float) -> float:
    """``D = exp(-r T)``. ``T <= 0`` → 1.0."""
    if time_to_expiry <= 0:
        return 1.0
    return math.exp(-continuous_rate * time_to_expiry)


def ex_date_instant_utc(ex_date: date) -> datetime:
    """Dividend drop modeled at 00:00 America/New_York on the ex-date → UTC."""
    local = datetime.combine(ex_date, time(0, 0), tzinfo=_ET)
    return local.astimezone(timezone.utc)


def years_to_ex_date(as_of_utc: datetime, ex_date: date) -> float:
    """ACT/365 year fraction from ``as_of_utc`` to ex-date midnight ET."""
    if as_of_utc.tzinfo is None:
        raise ValueError("as_of_utc must be timezone-aware")
    as_of = as_of_utc.astimezone(timezone.utc)
    ex_utc = ex_date_instant_utc(ex_date)
    seconds = (ex_utc - as_of).total_seconds()
    if seconds <= 0:
        return 0.0
    return seconds / _SECONDS_PER_ACT365_YEAR


def escrowed_forward(
    spot: float,
    continuous_rate: float,
    time_to_expiry: float,
    dividends: Sequence[Dividend],
    as_of_utc: datetime,
) -> tuple[float, tuple[Dividend, ...]]:
    """Escrowed-dividend forward ``F``.

    Parameters
    ----------
    spot:
        Underlying spot at trade (raw / unadjusted).
    continuous_rate:
        Continuous risk-free rate ``r`` consistent with ``D = exp(-r T)``.
    time_to_expiry:
        ``T`` in ACT/365 years (same ``T`` used by the solver).
    dividends:
        Static schedule for the underlying (already filtered by ticker).
    as_of_utc:
        Trade time (timezone-aware) for ``t_i`` measurement.

    Returns
    -------
    ``(F, dividends_applied)`` where ``dividends_applied`` are those with
    ``0 < t_i < T``.
    """
    if spot <= 0:
        raise ValueError("spot must be > 0")
    if time_to_expiry < 0:
        raise ValueError("time_to_expiry must be >= 0")
    if as_of_utc.tzinfo is None:
        raise ValueError("as_of_utc must be timezone-aware")

    if time_to_expiry == 0:
        return spot, ()

    r = continuous_rate
    t = time_to_expiry
    # Spot grown to expiry.
    f = spot * math.exp(r * t)
    applied: list[Dividend] = []
    for div in dividends:
        if div.amount < 0:
            raise ValueError(f"dividend amount must be >= 0, got {div.amount}")
        t_i = years_to_ex_date(as_of_utc, div.ex_date)
        # Strictly inside remaining life: after trade, before expiry.
        if t_i <= 0.0 or t_i >= t:
            continue
        # Future-value each cash amount from its own ex-date to expiry.
        f -= div.amount * math.exp(r * (t - t_i))
        applied.append(div)

    return f, tuple(applied)


def carry(
    spot: float,
    continuous_rate: float,
    as_of_utc: datetime,
    expiry_date: date,
    dividends: Sequence[Dividend],
    *,
    expiry_time_et: str = "16:00",
) -> CarryResult:
    """Compute ``T``, ``D``, and escrowed ``F`` for one trade."""
    t = years_to_expiry(as_of_utc, expiry_date, expiry_time_et=expiry_time_et)
    d = discount_factor(continuous_rate, t)
    f, applied = escrowed_forward(
        spot, continuous_rate, t, dividends, as_of_utc
    )
    return CarryResult(
        spot=spot,
        forward=f,
        discount=d,
        continuous_rate=continuous_rate,
        time_to_expiry=t,
        dividends_applied=applied,
    )
