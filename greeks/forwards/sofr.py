"""SOFR daily series → continuous rate for Black-76 discounting.

**Compounding convention (documented; keep stable under methodology_version):**

1. Overnight SOFR observations are treated as ACT/360 simple overnight rates
   (FRED series ``SOFR`` is percent; we store decimal, e.g. ``0.0433``).
2. For calendar days ``d`` in ``[as_of_date, expiry_date)`` (end exclusive),
   compound the **accumulation** factor (not a discount)::

       A = Π_d (1 + sofr_d / 360)

   Missing dates are **forward-filled** from the last available observation on
   or before ``d`` (weekends/holidays). If no prior observation exists, raise.
3. Continuous rate matching our ACT/365 solver ``T``::

       r = ln(A) / T_act365_calendar

   where ``T_act365_calendar = n_days / 365`` and ``n_days`` is the number of
   overnight steps (``r > 0`` when SOFR is positive). The solver then uses
   ``D = exp(-r * T_solver)`` with the second-resolution ``T`` from
   :func:`greeks.solver.time.years_to_expiry`.

Unit tests use the offline fixture under ``greeks/fixtures/sofr/`` — no FRED
or ClickHouse required for pure math.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import httpx

_FRED_SOFR_URL = "https://api.stlouisfed.org/fred/series/observations"
_DEFAULT_FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "sofr" / "sofr_daily.csv"
)


@dataclass(frozen=True)
class SofrObservation:
    """One daily SOFR fix (decimal annualized, not percent)."""

    obs_date: date
    rate: float  # e.g. 0.0433 for 4.33%
    source: str = "fixture"


def load_sofr_csv(path: Path | str | None = None) -> dict[date, float]:
    """Load ``date,rate`` CSV (rate as decimal). Offline unit-test path."""
    p = Path(path) if path is not None else _DEFAULT_FIXTURE
    if not p.is_file():
        raise FileNotFoundError(f"SOFR fixture not found: {p}")
    out: dict[date, float] = {}
    with p.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"SOFR CSV missing header: {p}")
        fields = {h.strip().lower() for h in reader.fieldnames}
        if "date" not in fields or "rate" not in fields:
            raise ValueError(f"SOFR CSV must have date,rate columns: {p}")
        for row in reader:
            d = date.fromisoformat(str(row["date"]).strip())
            rate = float(row["rate"])
            if rate < 0:
                raise ValueError(f"negative SOFR on {d}")
            out[d] = rate
    if not out:
        raise ValueError(f"SOFR CSV empty: {p}")
    return out


def rate_on_or_before(series: Mapping[date, float], d: date) -> float:
    """Forward-fill: last observation on or before ``d``."""
    if d in series:
        return series[d]
    prior = [k for k in series if k <= d]
    if not prior:
        raise KeyError(f"no SOFR observation on or before {d}")
    return series[max(prior)]


def compound_overnight_accumulation(
    series: Mapping[date, float],
    start: date,
    end: date,
) -> float:
    """Overnight accumulation ``A = Π (1 + sofr/360)`` for days in ``[start, end)``.

    If ``start == end``, returns ``1.0`` (no overnight steps). This is an
    **accumulation** factor (> 1 when rates are positive), not a discount.
    """
    if start > end:
        raise ValueError("start must be <= end")
    if start == end:
        return 1.0
    acc = 1.0
    d = start
    while d < end:
        s = rate_on_or_before(series, d)
        acc *= 1.0 + s / 360.0
        d += timedelta(days=1)
    return acc


# Back-compat alias (name was misleading; prefer compound_overnight_accumulation).
compound_overnight_df = compound_overnight_accumulation


def continuous_rate_from_sofr(
    series: Mapping[date, float],
    as_of_date: date,
    expiry_date: date,
) -> float:
    """Continuous ACT/365 rate from compounded overnight SOFR.

    ``r = ln(A) / (n/365)`` with ``n = (expiry_date - as_of_date).days`` and
    ``A`` the overnight accumulation. When ``n == 0``, returns the one-day
    continuous equivalent of the on-or-before SOFR fix (usable with intraday ``T``).
    """
    if expiry_date < as_of_date:
        raise ValueError("expiry_date must be >= as_of_date")
    n = (expiry_date - as_of_date).days
    if n == 0:
        s = rate_on_or_before(series, as_of_date)
        # One-day continuous equivalent under ACT/360 overnight → ACT/365 year.
        return 365.0 * math.log(1.0 + s / 360.0)
    acc = compound_overnight_accumulation(series, as_of_date, expiry_date)
    if acc <= 0:
        raise ValueError("compounded accumulation must be > 0")
    t_cal = n / 365.0
    return math.log(acc) / t_cal


# ── pre-SOFR fallback: the 1-month T-bill as an overnight rate ────────────────────────────────────
#
# SOFR begins 2018-04-02, but historical greeks price back to 2016. For a date before the first SOFR
# fix, the overnight rate is that date's 1-month Treasury-bill yield. Treasury yields are quoted
# ACT/365 (coupon/investment basis); the accrual above treats each rate as an ACT/360 simple overnight
# rate — one day contributes the factor (1 + rate/360). The same daily accrual expressed on the
# ACT/360 basis the accrual uses is therefore
#
#     r_360 = y_365 * 360 / 365
#
# so a converted bill rate enters compound_overnight_accumulation exactly like a SOFR fix.


def tbill_to_overnight_act360(bill_yield: float) -> float:
    """A 1-month T-bill yield (decimal, ACT/365) as an ACT/360 simple overnight rate: ``y * 360/365``.

    The SOFR accrual compounds ``(1 + rate/360)`` per day, so ``rate`` is an ACT/360-annualized simple
    rate; the Treasury yield is ACT/365-annualized, so the same daily rate is ``y * 360/365``."""
    return bill_yield * 360.0 / 365.0


def combined_overnight_series(
    sofr: Mapping[date, float],
    tbill_overnight: Mapping[date, float],
    first_sofr_date: date,
) -> dict[date, float]:
    """One overnight series spanning the SOFR boundary: the ACT/360 T-bill rate on dates strictly
    before ``first_sofr_date``, SOFR on dates on or after it. ``tbill_overnight`` is already
    ACT/360-converted (see :func:`tbill_to_overnight_act360`). An accrual window that crosses the
    boundary then compounds T-bills before it and SOFR after it, with no gap. T-bills never fill a
    date on or after ``first_sofr_date`` — a missing SOFR fix there stays missing (an error upstream).
    """
    out = {d: r for d, r in tbill_overnight.items() if d < first_sofr_date}
    out.update({d: r for d, r in sofr.items() if d >= first_sofr_date})
    return out


def discount_from_sofr(
    series: Mapping[date, float],
    as_of_date: date,
    expiry_date: date,
    time_to_expiry: float,
) -> tuple[float, float]:
    """Return ``(D, r)`` with ``D = exp(-r * time_to_expiry)``."""
    r = continuous_rate_from_sofr(series, as_of_date, expiry_date)
    if time_to_expiry <= 0:
        return 1.0, r
    return math.exp(-r * time_to_expiry), r


def fetch_fred_sofr(
    api_key: str,
    *,
    observation_start: date,
    observation_end: date,
    timeout_s: float = 60.0,
) -> list[SofrObservation]:
    """Sync FRED pull of daily SOFR (no asyncio). Returns decimal rates.

    Does not write ClickHouse — caller inserts via :func:`rows_for_clickhouse`.
    """
    if not api_key:
        raise ValueError("FRED API key is required")
    params = {
        "series_id": "SOFR",
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start.isoformat(),
        "observation_end": observation_end.isoformat(),
    }
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.get(_FRED_SOFR_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
    obs_raw = payload.get("observations")
    if not isinstance(obs_raw, list):
        raise ValueError("FRED response missing observations list")
    out: list[SofrObservation] = []
    for item in obs_raw:
        if not isinstance(item, Mapping):
            continue
        val = str(item.get("value", ".")).strip()
        if val in (".", ""):
            continue
        # FRED SOFR is percent (e.g. 4.33); store decimal.
        rate = float(val) / 100.0
        out.append(
            SofrObservation(
                obs_date=date.fromisoformat(str(item["date"])),
                rate=rate,
                source="fred",
            )
        )
    return out


def rows_for_clickhouse(
    observations: Sequence[SofrObservation],
    *,
    fetched_at: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Map observations to dict rows for ``trading.sofr_daily`` insert."""
    ts = fetched_at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        raise ValueError("fetched_at must be timezone-aware")
    ts = ts.astimezone(timezone.utc)
    return [
        {
            "date": o.obs_date,
            "rate": o.rate,
            "source": o.source,
            "fetched_at": ts,
        }
        for o in observations
    ]


def series_from_observations(
    observations: Sequence[SofrObservation],
) -> dict[date, float]:
    """Build ``date -> rate`` map (last write wins if duplicates)."""
    return {o.obs_date: o.rate for o in observations}
