"""The optimizer's edge is the gamma form of the objective spec §2 ("Refined 2026-11-19"):

    edge = −½ · γ · S² · (σ_IV² − σ_RV²) / 365   (dollars/calendar-day per unit)

γ is position-signed as CalcServer sends it (short negative), so a short with iv>rv earns
positive edge and a long the negative. These tests pin the exact value and sign, show the gamma
form coincides with the old theta form when theta is the zero-rate Black-76 theta, and show that
a put and a call with equal gamma now get equal edge (the theta form gave them different edge
because Black-76 theta carries a rate term that differs by right).
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import optimizer  # noqa: E402

_CAL = optimizer.CALENDAR_DAYS_PER_YEAR  # 365.0

_POLICY = {"bp_limit_pct": 60, "bp_reserve_pct": 0, "risk_aversion": 0.0,
           "hedge_interval_days": 1.0, "min_improvement": 0.0}
_CAPITAL = {"net_liq": 100_000.0}


def _option(uid: str, *, right: str, gamma: float, iv: float, rv: float,
            theta: float = 0.0, spot: float = 100.0) -> dict:
    return {
        "id": f"{uid}260101{right}00100000-short", "underlying": uid, "type": "option",
        "direction": "short", "current_quantity": 0,
        "theta_per_unit": theta, "exec_cost_per_unit": 0.0,
        "delta_per_unit": 0.0, "gamma_per_unit": gamma, "margin_per_unit": 100.0,
        "iv": iv, "rv_forecast": rv, "days_to_expiry": 5.0, "strike": 100.0, "expiry": "2026-01-01",
    }


def _und(uid: str, spot: float = 100.0) -> dict:
    return {"underlying": uid, "spot": spot, "delta_target_pct": 0.0,
            "delta_tolerance_pct": 10.0, "tradeable": True, "sigma_rv": 0.3}


def _edge(dvs: list, unds: list) -> list:
    groups = optimizer._group_by_underlying(dvs)
    bundle = optimizer._extract_inputs(dvs, unds, _POLICY, _CAPITAL, groups, horizon_days=5.0)
    return list(bundle["edge"])


def _gamma_edge(gamma: float, spot: float, iv: float, rv: float) -> float:
    return -0.5 * gamma * spot ** 2 * (iv ** 2 - rv ** 2) / _CAL


# ── exact value and sign: a short earns positive edge, a long negative ───────────────────────────

def test_short_and_long_exact_edge() -> None:
    spot, iv, rv = 100.0, 0.3, 0.2
    short = _option("SHORT", right="C", gamma=-2.0, iv=iv, rv=rv)   # short: γ < 0
    long_ = _option("LONG", right="C", gamma=+2.0, iv=iv, rv=rv)    # long:  γ > 0
    edge = _edge([short, long_], [_und("SHORT"), _und("LONG")])

    expected = _gamma_edge(-2.0, spot, iv, rv)  # = 500/365
    assert edge[0] == pytest.approx(expected, abs=1e-12)
    assert edge[0] == pytest.approx(500.0 / 365.0, abs=1e-12)
    assert edge[0] > 0.0                         # short with iv>rv -> positive edge
    assert edge[1] == pytest.approx(-expected, abs=1e-12)
    assert edge[1] < 0.0                         # long -> negative edge


# ── at zero rates the gamma form equals the old theta form ───────────────────────────────────────

def test_zero_rate_theta_form_matches_gamma_form() -> None:
    """With consistent Black-76 greeks at r = q = 0, theta = −½γS²σ_IV²/365, and then the old
    theta form  edge = theta·(1 − (rv/iv)²)  is algebraically the gamma form. Pin it to 1e-9."""
    gamma, spot, iv, rv = -1.5, 100.0, 0.4, 0.25
    theta_zero_rate = -0.5 * gamma * spot ** 2 * iv ** 2 / _CAL   # Black-76 theta at r=q=0
    old_form = theta_zero_rate * (1.0 - (rv / iv) ** 2)

    opt = _option("ZR", right="C", gamma=gamma, iv=iv, rv=rv, theta=theta_zero_rate)
    edge = _edge([opt], [_und("ZR")])

    assert edge[0] == pytest.approx(_gamma_edge(gamma, spot, iv, rv), abs=1e-12)
    assert edge[0] == pytest.approx(old_form, abs=1e-9)


# ── a put and a call with equal gamma now get equal edge (the theta form did not) ────────────────

def test_put_and_call_equal_gamma_get_equal_edge() -> None:
    """Same strike/spot/iv/rv and equal gamma, but Black-76 theta differs by right under r>0.
    The gamma form gives them identical edge; the old theta form would not."""
    gamma, spot, iv, rv = 1.8, 100.0, 0.35, 0.22
    theta_call, theta_put = -20.0, -18.0   # differ under r>0 (the carry term differs by right)

    call = _option("CALLU", right="C", gamma=gamma, iv=iv, rv=rv, theta=theta_call)
    put = _option("PUTU", right="P", gamma=gamma, iv=iv, rv=rv, theta=theta_put)
    edge = _edge([call, put], [_und("CALLU"), _und("PUTU")])

    # Gamma form: identical.
    assert edge[0] == pytest.approx(edge[1], abs=1e-12)
    assert edge[0] == pytest.approx(_gamma_edge(gamma, spot, iv, rv), abs=1e-12)

    # Old theta form: would have differed, because theta differs by right.
    old_call = theta_call * (1.0 - (rv / iv) ** 2)
    old_put = theta_put * (1.0 - (rv / iv) ** 2)
    assert abs(old_call - old_put) > 0.1
