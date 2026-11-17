"""The weighted isotonic fit, horizon interpolation and support rules (method iso_v1)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import List

from iv_curve import KIND_EXPIRY, KIND_HORIZON, STATUS_BEYOND_LAST_EXPIRY, STATUS_OK
from iv_curve.iso import CurvePoint, ExpiryInput, build_curve, pava


def _expiry(tau: int) -> date:
    # A distinct date per τ (exact value irrelevant to the fit; only ordering/identity matters).
    return date(2026, 1, 1) + timedelta(days=tau)


def _in(tau: int, var: float, w: float) -> ExpiryInput:
    return ExpiryInput(expiry=_expiry(tau), tau=tau, raw_total_var=var, n_trades=w)


def _expiries(rows: List[CurvePoint]) -> List[CurvePoint]:
    return [r for r in rows if r.kind == KIND_EXPIRY]


def _horizon(rows: List[CurvePoint], h: int) -> CurvePoint:
    return next(r for r in rows if r.kind == KIND_HORIZON and r.horizon_td == h)


# ── the PAVA primitive ─────────────────────────────────────────────────────────────────────────────

def test_pava_monotone_input_unchanged() -> None:
    fit, support = pava([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    assert fit == [1.0, 2.0, 3.0]
    assert support == [1.0, 1.0, 1.0]


def test_pava_single_violation_pools_to_weighted_mean() -> None:
    # 3 (w=10) then 1 (w=30): a decrease, so pool to (3·10 + 1·30)/40 = 1.5; support is the sum, 40.
    fit, support = pava([3.0, 1.0], [10.0, 30.0])
    assert fit == [1.5, 1.5]
    assert support == [40.0, 40.0]


# ── the fit over expiries ──────────────────────────────────────────────────────────────────────────

def test_already_monotone_fit_returned_unchanged() -> None:
    rows = build_curve([_in(5, 0.01, 10), _in(10, 0.02, 10), _in(21, 0.05, 10)])
    for r in _expiries(rows):
        assert r.status == STATUS_OK
        assert r.fit_total_var == r.raw_total_var  # nothing to pool — fit equals raw


def test_single_violation_pools_to_weighted_mean() -> None:
    rows = _expiries(build_curve([_in(5, 0.03, 10), _in(10, 0.01, 30)]))
    assert [r.fit_total_var for r in rows] == [0.015, 0.015]  # (0.03·10 + 0.01·30)/40
    assert [r.support for r in rows] == [40.0, 40.0]          # block support = Σ n_trades


def test_first_window_jump_survives_unchanged() -> None:
    # A large jump from the anchor (0) to the first expiry is an INCREASE, which isotonic never
    # flattens — the EWZ first-week event must be preserved in the fit.
    rows = _expiries(build_curve([_in(5, 0.08, 10), _in(21, 0.10, 10)]))
    assert [r.fit_total_var for r in rows] == [0.08, 0.10]


# ── horizon interpolation, brackets and support ─────────────────────────────────────────────────────

def test_anchor_bracket_and_on_node_and_beyond() -> None:
    e10 = _expiry(10)
    rows = build_curve([_in(10, 0.02, 20)])

    h5 = _horizon(rows, 5)   # between the anchor (0,0) and the single expiry at τ=10
    assert h5.status == STATUS_OK
    assert h5.fit_total_var == 0.01            # 0 + 0.02·(5/10)
    assert h5.support == 20.0                  # min(anchor=∞, 20)
    assert h5.bracket_lo_expiry is None        # anchor is the lower bracket
    assert h5.bracket_hi_expiry == e10

    h10 = _horizon(rows, 10)                   # lands exactly on the expiry node
    assert h10.status == STATUS_OK
    assert h10.fit_total_var == 0.02
    assert h10.support == 20.0
    assert h10.bracket_lo_expiry == e10
    assert h10.bracket_hi_expiry == e10

    for h in (21, 42):                         # past the last expiry — no extrapolation
        row = _horizon(rows, h)
        assert row.status == STATUS_BEYOND_LAST_EXPIRY
        assert row.fit_total_var is None
        assert row.support is None
        assert row.bracket_lo_expiry is None and row.bracket_hi_expiry is None


def test_support_is_the_smaller_bracket() -> None:
    e10, e42 = _expiry(10), _expiry(42)
    rows = build_curve([_in(10, 0.02, 100), _in(42, 0.05, 5)])

    h21 = _horizon(rows, 21)                   # between τ=10 (support 100) and τ=42 (support 5)
    assert h21.status == STATUS_OK
    assert abs(h21.fit_total_var - (0.02 + 0.03 * (21 - 10) / (42 - 10))) < 1e-12
    assert h21.support == 5.0                  # the smaller of the two brackets
    assert h21.bracket_lo_expiry == e10
    assert h21.bracket_hi_expiry == e42

    assert _horizon(rows, 5).support == 100.0  # anchor→τ10: min(∞, 100)


def test_no_ok_expiry_gives_no_rows() -> None:
    assert build_curve([]) == []
