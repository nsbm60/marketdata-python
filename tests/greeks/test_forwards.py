"""PR3: escrowed forward + SOFR discount (offline fixture, no network)."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from greeks.domain import Dividend
from greeks.forwards.carry import (
    carry,
    discount_factor,
    escrowed_forward,
    years_to_ex_date,
)
from greeks.forwards.sofr import (
    compound_overnight_accumulation,
    continuous_rate_from_sofr,
    discount_from_sofr,
    load_sofr_csv,
    rate_on_or_before,
)
from greeks.solver.time import years_to_expiry

_ET = ZoneInfo("America/New_York")


def test_load_sofr_fixture() -> None:
    series = load_sofr_csv()
    assert len(series) >= 20
    # Weekday published rate
    assert abs(series[date(2026, 5, 1)] - 0.0433) < 1e-15


def test_sofr_forward_fill_weekend() -> None:
    series = load_sofr_csv()
    # 2026-05-02 is Saturday — forward-fill from Friday 2026-05-01
    sat = date(2026, 5, 2)
    assert sat not in series
    assert rate_on_or_before(series, sat) == series[date(2026, 5, 1)]


def test_constant_sofr_continuous_rate() -> None:
    series = load_sofr_csv()
    start = date(2026, 5, 27)
    end = date(2026, 6, 26)  # 30 calendar days
    n = (end - start).days
    assert n == 30
    acc = compound_overnight_accumulation(series, start, end)
    expected_acc = (1.0 + 0.0433 / 360.0) ** n
    assert abs(acc - expected_acc) < 1e-12
    r = continuous_rate_from_sofr(series, start, end)
    assert r > 0
    assert abs(r - (math.log(expected_acc) / (n / 365.0))) < 1e-12


def test_discount_from_sofr_matches_exp() -> None:
    series = load_sofr_csv()
    as_of = date(2026, 5, 27)
    expiry = date(2026, 6, 26)
    t = 30.0 / 365.0
    d, r = discount_from_sofr(series, as_of, expiry, t)
    assert abs(d - math.exp(-r * t)) < 1e-15


def test_discount_factor_zero_t() -> None:
    assert discount_factor(0.05, 0.0) == 1.0


def test_escrowed_forward_no_div() -> None:
    as_of = datetime(2026, 5, 27, 14, 30, tzinfo=_ET).astimezone(timezone.utc)
    s, r, t = 100.0, 0.05, 30.0 / 365.0
    f, applied = escrowed_forward(s, r, t, (), as_of)
    assert applied == ()
    assert abs(f - s * math.exp(r * t)) < 1e-12


def test_escrowed_forward_nvda_div_inside_life() -> None:
    """NVDA $0.25 ex 2026-06-04 between trade 2026-05-28 and expiry 2026-06-20."""
    as_of = datetime(2026, 5, 28, 15, 0, tzinfo=_ET).astimezone(timezone.utc)
    expiry = date(2026, 6, 20)
    t = years_to_expiry(as_of, expiry)
    assert t > 0
    div = Dividend(underlying="NVDA", amount=0.25, ex_date=date(2026, 6, 4))
    s, r = 120.0, 0.043
    f, applied = escrowed_forward(s, r, t, (div,), as_of)
    assert applied == (div,)
    t_i = years_to_ex_date(as_of, div.ex_date)
    assert 0 < t_i < t
    expected = s * math.exp(r * t) - div.amount * math.exp(r * (t - t_i))
    assert abs(f - expected) < 1e-12


def test_escrowed_rejects_pv_then_grow_identity() -> None:
    """Numerically show escrowed form differs from PV-then-grow when t_i != 0.

    For a single dividend the two forms coincide algebraically when the same
    r and t_i are used for PV and growth — but multi-date schedules and the
    single-date FV-from-trade variant differ. Guard the **single-date FV from
    trade** anti-pattern: F = S e^{rT} - div e^{rT} (wrong).
    """
    as_of = datetime(2026, 5, 28, 15, 0, tzinfo=_ET).astimezone(timezone.utc)
    t = 40.0 / 365.0
    div = Dividend(underlying="NVDA", amount=0.25, ex_date=date(2026, 6, 4))
    s, r = 120.0, 0.05
    f, _ = escrowed_forward(s, r, t, (div,), as_of)
    wrong_single_date_fv = s * math.exp(r * t) - div.amount * math.exp(r * t)
    assert abs(f - wrong_single_date_fv) > 1e-6


def test_dividend_before_trade_excluded() -> None:
    as_of = datetime(2026, 6, 10, 15, 0, tzinfo=_ET).astimezone(timezone.utc)
    t = 20.0 / 365.0
    div = Dividend(underlying="NVDA", amount=0.25, ex_date=date(2026, 6, 4))
    f, applied = escrowed_forward(100.0, 0.05, t, (div,), as_of)
    assert applied == ()
    assert abs(f - 100.0 * math.exp(0.05 * t)) < 1e-12


def test_dividend_after_expiry_excluded() -> None:
    as_of = datetime(2026, 5, 28, 15, 0, tzinfo=_ET).astimezone(timezone.utc)
    t = 5.0 / 365.0  # expiry ~ Jun 2; div Jun 4 is after
    div = Dividend(underlying="NVDA", amount=0.25, ex_date=date(2026, 6, 4))
    t_i = years_to_ex_date(as_of, div.ex_date)
    assert t_i > t
    f, applied = escrowed_forward(100.0, 0.05, t, (div,), as_of)
    assert applied == ()


def test_carry_end_to_end_with_sofr() -> None:
    series = load_sofr_csv()
    as_of_local = datetime(2026, 5, 28, 10, 30, tzinfo=_ET)
    as_of = as_of_local.astimezone(timezone.utc)
    expiry = date(2026, 6, 20)
    r = continuous_rate_from_sofr(series, as_of_local.date(), expiry)
    divs = (Dividend(underlying="NVDA", amount=0.25, ex_date=date(2026, 6, 4)),)
    result = carry(135.0, r, as_of, expiry, divs)
    assert result.time_to_expiry > 0
    assert 0 < result.discount < 1
    assert result.forward > 0
    assert result.dividends_applied == divs
    # D = exp(-r T)
    assert abs(result.discount - math.exp(-r * result.time_to_expiry)) < 1e-15
    # F < S e^{rT} when a positive div is escrowed
    assert result.forward < 135.0 * math.exp(r * result.time_to_expiry)


def test_mu_dividend_applied_when_ex_inside_life() -> None:
    """MU $0.15 ex 2026-07-06 (board record date; T+1 ex ≈ record)."""
    as_of = datetime(2026, 6, 20, 15, 0, tzinfo=_ET).astimezone(timezone.utc)
    expiry = date(2026, 7, 17)
    t = years_to_expiry(as_of, expiry)
    div = Dividend(underlying="MU", amount=0.15, ex_date=date(2026, 7, 6))
    s, r = 120.0, 0.04
    f, applied = escrowed_forward(s, r, t, (div,), as_of)
    assert applied == (div,)
    t_i = years_to_ex_date(as_of, div.ex_date)
    expected = s * math.exp(r * t) - div.amount * math.exp(r * (t - t_i))
    assert abs(f - expected) < 1e-12
