"""Offline tests for the rv_daily intraday measures reference (rv/measures.py) — no ClickHouse."""

import math

from rv import measures
from rv.measures import MU


def test_hand_computed_small_case():
    # Open 100, close up to 101, back to 100. r1 = log(1.01), r2 = -r1. Overnight flat.
    r1 = math.log(1.01)
    m = measures.day_measures([100.0, 101.0, 100.0], 0.0)
    assert math.isclose(m.rv_intraday, 2 * r1 * r1)
    # r2 < 0 -> rs_down; r1 > 0 -> rs_up; each is exactly r1².
    assert math.isclose(m.rs_down, r1 * r1)
    assert math.isclose(m.rs_up, r1 * r1)
    # n = 2: bv = (π/2)·(2/1)·|r2|·|r1| = π·r1². tq is None (n < 3).
    assert math.isclose(m.bv_intraday, math.pi * r1 * r1)
    assert m.tq_intraday is None


def test_semivariance_splits_rv_total():
    # rs_down + rs_up must equal rv_intraday + overnight² (= rv_total) for an arbitrary day.
    grid = [100.0, 101.3, 100.7, 102.1, 101.5, 101.9]
    o = -0.012
    m = measures.day_measures(grid, o)
    assert math.isclose(m.rs_down + m.rs_up, m.rv_intraday + o * o, rel_tol=0, abs_tol=1e-15)


def test_zero_returns_go_in_neither_bucket():
    # Flat step (r1 = 0) then an up move: the zero return contributes to neither semivariance.
    grid = [100.0, 100.0, 101.0]
    r = math.log(1.01)
    m = measures.day_measures(grid, 0.0)
    assert m.rs_down == 0.0
    assert math.isclose(m.rs_up, r * r)
    assert math.isclose(m.rv_intraday, r * r)


def test_constant_prices_all_zero():
    m = measures.day_measures([100.0, 100.0, 100.0, 100.0], 0.0)
    assert m.rv_intraday == 0.0
    assert m.rs_down == 0.0
    assert m.rs_up == 0.0
    # bv/tq are defined (n = 3) and exactly zero — not None, not defaulted.
    assert m.bv_intraday == 0.0
    assert m.tq_intraday == 0.0


def test_null_rules_by_n():
    # n = 1: both bv and tq undefined.
    m1 = measures.day_measures([100.0, 101.0], 0.0)
    assert m1.bv_intraday is None and m1.tq_intraday is None
    # n = 2: bv defined, tq undefined.
    m2 = measures.day_measures([100.0, 101.0, 102.0], 0.0)
    assert m2.bv_intraday is not None and m2.tq_intraday is None
    # n = 3: both defined.
    m3 = measures.day_measures([100.0, 101.0, 102.0, 103.0], 0.0)
    assert m3.bv_intraday is not None and m3.tq_intraday is not None


def test_one_jump_bipower_well_below_rv():
    # A near-flat day with one large jump: rv² is dominated by the jump, but bipower pairs the jump
    # with a tiny neighbour, so bv stays well below rv — the jump signature the Stage 3b test uses.
    grid = [100.0, 100.01, 100.02, 100.03, 130.0, 130.01, 130.02]
    m = measures.day_measures(grid, 0.0)
    assert m.bv_intraday is not None
    assert m.bv_intraday < 0.5 * m.rv_intraday


def test_overnight_sign_routes_to_one_side():
    # Flat intraday so only the overnight return contributes; its sign picks the bucket.
    flat = [100.0, 100.0, 100.0, 100.0]
    down = measures.day_measures(flat, -0.02)
    assert math.isclose(down.rs_down, 0.02 * 0.02) and down.rs_up == 0.0
    up = measures.day_measures(flat, 0.03)
    assert math.isclose(up.rs_up, 0.03 * 0.03) and up.rs_down == 0.0


def test_mu_constant_value():
    # μ = 2^(2/3)·Γ(7/6)/Γ(1/2) ≈ 0.8309.
    assert math.isclose(MU, 0.8308609, abs_tol=1e-6)
