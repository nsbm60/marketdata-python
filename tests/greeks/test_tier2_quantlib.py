"""Tier-2: QuantLib Black-76 cross-check against py_vollib + Tier-1 fixtures.

Skipped automatically when QuantLib is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ql = pytest.importorskip("QuantLib")

from greeks.domain import OptionRight  # noqa: E402
from greeks.solver import black76 as ours  # noqa: E402
from greeks.solver import quantlib_black as ql_black  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "greeks" / "fixtures" / "v1"
IV_TOL = 1e-8
GREEK_TOL = 1e-6
PRICE_TOL = 1e-10


def _load_rows() -> list[dict[str, object]]:
    path = FIXTURE_DIR / "grid.jsonl"
    if not path.is_file():
        pytest.skip(f"fixtures not generated: {path}")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _right(row: dict[str, object]) -> OptionRight:
    return OptionRight(str(row["right"]))


def test_quantlib_importable() -> None:
    assert ql is not None
    assert hasattr(ql, "blackFormula")


def test_single_point_price_and_greeks_match() -> None:
    """Sanity ATM point: price + unit-aligned greeks within tolerance."""
    F, K, T, r, sigma = 100.0, 100.0, 30.0 / 365.0, 0.05, 0.25
    D = __import__("math").exp(-r * T)
    right = OptionRight.CALL
    spot = F

    px_ours = ours.price(F, K, T, D, sigma, right)
    px_ql = ql_black.price(F, K, T, D, sigma, right)
    assert abs(px_ours - px_ql) <= PRICE_TOL

    g_ours = ours.greeks(F, K, T, D, sigma, right, spot=spot)
    g_ql = ql_black.greeks(F, K, T, D, sigma, right, spot=spot)
    assert abs(g_ours.delta - g_ql.delta) <= GREEK_TOL
    assert abs(g_ours.gamma - g_ql.gamma) <= GREEK_TOL
    assert abs(g_ours.vega - g_ql.vega) <= GREEK_TOL
    assert abs(g_ours.theta - g_ql.theta) <= GREEK_TOL
    # Futures-style rho (documented); not QuantLib cash rho
    assert abs(g_ours.rho - g_ql.rho_futures) <= GREEK_TOL
    assert abs(g_ours.rho - g_ql.rho_cash) > GREEK_TOL  # systematic difference


def test_tier2_grid_price_iv_greeks() -> None:
    """Full Tier-1 fixture grid through QuantLib."""
    rows = _load_rows()
    assert len(rows) > 100

    worst_iv_rel = 0.0
    worst_greeks = {
        "delta": 0.0,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": 0.0,
        "rho_futures": 0.0,
        "price": 0.0,
    }

    for row in rows:
        right = _right(row)
        F = float(row["forward"])  # type: ignore[arg-type]
        K = float(row["strike"])  # type: ignore[arg-type]
        T = float(row["t"])  # type: ignore[arg-type]
        D = float(row["discount"])  # type: ignore[arg-type]
        sigma = float(row["true_vol"])  # type: ignore[arg-type]
        spot = float(row["spot"])  # type: ignore[arg-type]
        fixture_price = float(row["price"])  # type: ignore[arg-type]
        expected_iv = float(row["expected_iv"])  # type: ignore[arg-type]

        px_ql = ql_black.price(F, K, T, D, sigma, right)
        px_ours = ours.price(F, K, T, D, sigma, right)
        worst_greeks["price"] = max(worst_greeks["price"], abs(px_ql - px_ours))
        assert abs(px_ql - px_ours) <= PRICE_TOL, row
        assert abs(px_ql - fixture_price) <= PRICE_TOL, row

        iv_ql = ql_black.implied_vol(fixture_price, F, K, T, D, right)
        rel = abs(iv_ql - sigma) / sigma
        worst_iv_rel = max(worst_iv_rel, rel)
        assert rel <= IV_TOL, f"rel_iv={rel} row={row}"
        assert abs(iv_ql - expected_iv) <= IV_TOL

        g_ours = ours.greeks(F, K, T, D, sigma, right, spot=spot)
        g_ql = ql_black.greeks(F, K, T, D, sigma, right, spot=spot)
        for name, a, b in (
            ("delta", g_ours.delta, g_ql.delta),
            ("gamma", g_ours.gamma, g_ql.gamma),
            ("vega", g_ours.vega, g_ql.vega),
            ("theta", g_ours.theta, g_ql.theta),
            ("rho_futures", g_ours.rho, g_ql.rho_futures),
        ):
            diff = abs(a - b)
            worst_greeks[name] = max(worst_greeks[name], diff)
            assert diff <= GREEK_TOL, f"{name} diff={diff} row={row}"

    assert worst_iv_rel <= IV_TOL
    for name, w in worst_greeks.items():
        if name == "price":
            assert w <= PRICE_TOL
        else:
            assert w <= GREEK_TOL


def test_rho_cash_differs_from_methodology() -> None:
    """Documented systematic: QL cash rho ≠ py_vollib F-fixed rho."""
    F, K, T, r, sigma = 100.0, 100.0, 30.0 / 365.0, 0.05, 0.25
    D = __import__("math").exp(-r * T)
    g_ours = ours.greeks(F, K, T, D, sigma, OptionRight.CALL, spot=F)
    g_ql = ql_black.greeks(F, K, T, D, sigma, OptionRight.CALL, spot=F)
    # Cash rho is a different economic quantity — must not silently match
    assert abs(g_ours.rho - g_ql.rho_cash) > 1e-3
    assert abs(g_ours.rho - g_ql.rho_futures) <= GREEK_TOL
