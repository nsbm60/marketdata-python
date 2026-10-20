"""Tier-1 v2 fixtures: spot-terms gamma, gamma_S = gamma_F * (F/S)**2.

The v1 grid priced every row with S = F, so it can't see the conversion. The v2
grid has S = 100 and F = S*exp(rT), so F/S > 1 on every row.

IV and greeks are compared to the frozen values with a tight relative tolerance
rather than bit for bit: the last ulp of Jäckel's IV differs across platforms
(the v1 bit-for-bit check fails by one ulp on linux-aarch64).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from greeks.domain import OptionRight, SolverInput
from greeks.solver.black76 import greeks as black_greeks
from greeks.solver.black76 import price as black_price
from greeks.solver.invert import invert, relative_iv_error

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "greeks" / "fixtures" / "v2"
MAX_REL = 1e-10
FROZEN_REL = 1e-12


def _load_rows() -> list[dict[str, object]]:
    path = FIXTURE_DIR / "grid.jsonl"
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _f(row: dict[str, object], key: str) -> float:
    return float(row[key])  # type: ignore[arg-type]


def _close(a: float, b: float, rel: float) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300) or abs(a - b) <= 1e-300


def test_meta() -> None:
    meta = json.loads((FIXTURE_DIR / "META.json").read_text(encoding="utf-8"))
    assert meta["methodology_version"] == "v2"
    assert meta["max_relative_iv_error"] == MAX_REL


def test_grid_has_spot_below_forward() -> None:
    rows = _load_rows()
    assert len(rows) > 100
    assert all(_f(r, "forward") > _f(r, "spot") for r in rows)


def test_v2_recovery_and_frozen_greeks() -> None:
    for row in _load_rows():
        result = invert(
            SolverInput(
                trade_price=_f(row, "price"),
                strike=_f(row, "strike"),
                right=OptionRight(str(row["right"])),
                forward=_f(row, "forward"),
                discount=_f(row, "discount"),
                time_to_expiry=_f(row, "t"),
                spot=_f(row, "spot"),
            ),
            methodology_version="v2",
            t_floor_minutes=0.0,
        )
        assert result.status.value == "success", row
        assert result.iv is not None
        assert relative_iv_error(result.iv, _f(row, "true_vol")) <= MAX_REL, row
        assert _close(result.iv, _f(row, "expected_iv"), FROZEN_REL), row
        for name in ("delta", "gamma", "vega", "theta", "rho"):
            got = getattr(result, name)
            assert got is not None
            # Greeks inherit the IV's last-ulp noise; 1e-9 relative is far below
            # any methodology change (gamma moves by (F/S)^2 - 1 >= 1e-6 here).
            assert _close(got, _f(row, f"expected_{name}"), 1e-9), (name, row)


def test_gamma_is_forward_gamma_times_f_over_s_squared() -> None:
    for row in _load_rows()[::7]:
        args = (
            _f(row, "forward"),
            _f(row, "strike"),
            _f(row, "t"),
            _f(row, "discount"),
            _f(row, "true_vol"),
            OptionRight(str(row["right"])),
        )
        spot = _f(row, "spot")
        fwd = black_greeks(*args)
        spt = black_greeks(*args, spot=spot)
        ratio = args[0] / spot
        assert fwd.delta_is_spot is False
        assert spt.delta_is_spot is True
        assert spt.gamma == pytest.approx(fwd.gamma * ratio * ratio, rel=1e-15)
        assert spt.delta == pytest.approx(fwd.delta * ratio, rel=1e-15)


def _value_at_spot(s: float, row: dict[str, object]) -> float:
    """Black-76 value with the forward carried from spot: F = S*exp(rT), no dividends."""
    t = _f(row, "t")
    r = _f(row, "r")
    return black_price(
        s * math.exp(r * t),
        _f(row, "strike"),
        t,
        _f(row, "discount"),
        _f(row, "true_vol"),
        OptionRight(str(row["right"])),
    )


def test_spot_gamma_matches_finite_difference() -> None:
    """d2V/dS2 by central difference agrees with the published spot gamma.

    Rows with T >= 10 days and near-money strikes. There (F/S)^2 - 1 >= 2.7e-3, so
    the 1e-4 tolerance (finite-difference error) still rejects forward gamma.
    """
    checked = 0
    for row in _load_rows():
        t = _f(row, "t")
        m = _f(row, "moneyness")
        if t * 365.0 < 10.0 or not 0.9 <= m <= 1.1:
            continue
        s = _f(row, "spot")
        h = 1e-4 * s
        v_up = _value_at_spot(s + h, row)
        v_0 = _value_at_spot(s, row)
        v_dn = _value_at_spot(s - h, row)
        fd_gamma = (v_up - 2.0 * v_0 + v_dn) / (h * h)
        fd_delta = (v_up - v_dn) / (2.0 * h)
        g = black_greeks(
            _f(row, "forward"),
            _f(row, "strike"),
            t,
            _f(row, "discount"),
            _f(row, "true_vol"),
            OptionRight(str(row["right"])),
            spot=s,
        )
        assert g.gamma == pytest.approx(fd_gamma, rel=1e-4, abs=1e-9), row
        assert g.delta == pytest.approx(fd_delta, rel=1e-4, abs=1e-6), row
        checked += 1
    assert checked > 100
