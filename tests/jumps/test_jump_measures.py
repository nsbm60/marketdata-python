"""Offline tests for the jump_daily measures reference (jumps/measures.py) — no ClickHouse."""

import math

from jumps import ON_MIN_HISTORY, THETA, Z_INTRADAY, Z_OVERNIGHT, measures


# ── intraday jump z and flags ─────────────────────────────────────────────────────────────────────

def test_no_jump_day_has_small_z_and_zero_flag():
    # BV ≈ RV (continuous day): (RV−BV)/RV ≈ 0 → z ≈ 0 → not a jump; cont_var = RV, jump_var = 0.
    rv, bv, tq, n = 1.0e-4, 0.99e-4, 1.0e-8, 78
    z = measures.intraday_jump_z(rv, bv, tq, n)
    assert z is not None and z < Z_INTRADAY
    flag = measures.intraday_flag(z)
    assert flag == 0
    jump_var, cont_var = measures.jump_cont_var(rv, bv, flag)
    assert jump_var == 0.0 and math.isclose(cont_var, rv)


def test_one_jump_day_crosses_threshold_and_splits_variance():
    # RV far above BV (a jump dominates RV but not the bipower product) → large z → flagged.
    rv, bv, tq, n = 5.0e-4, 1.0e-4, 1.0e-8, 78
    z = measures.intraday_jump_z(rv, bv, tq, n)
    assert z is not None and z > Z_INTRADAY
    flag = measures.intraday_flag(z)
    assert flag == 1
    jump_var, cont_var = measures.jump_cont_var(rv, bv, flag)
    assert math.isclose(jump_var, rv - bv)                 # max(RV−BV, 0)
    assert math.isclose(cont_var, bv)                      # RV − jump_var = BV
    assert math.isclose(jump_var + cont_var, rv)           # sums to RV


def test_jump_z_null_rules():
    assert measures.intraday_jump_z(1e-4, None, 1e-8, 78) is None   # BV missing
    assert measures.intraday_jump_z(1e-4, 1e-4, None, 78) is None   # TQ missing
    assert measures.intraday_jump_z(0.0, 1e-4, 1e-8, 78) is None    # RV = 0
    assert measures.intraday_jump_z(1e-4, 0.0, 1e-8, 78) is None    # BV = 0
    assert measures.intraday_flag(None) is None
    assert measures.jump_cont_var(1e-4, None, None) == (None, None)


def test_tq_over_bv2_branch_floors_at_one():
    # TQ/BV² < 1 → max(1, ·) = 1, so the scale is sqrt(θ/n) and z = ratio / sqrt(θ/n).
    rv, bv, n = 2.0e-4, 1.0e-4, 50
    tq_small = 0.5 * bv * bv            # TQ/BV² = 0.5 < 1 → floored to 1
    z = measures.intraday_jump_z(rv, bv, tq_small, n)
    expected = ((rv - bv) / rv) / math.sqrt(THETA * (1.0 / n) * 1.0)
    assert math.isclose(z, expected)
    # TQ/BV² = 4 > 1 → the branch uses 4, giving a smaller z than the floored case.
    z_big = measures.intraday_jump_z(rv, bv, 4.0 * bv * bv, n)
    assert z_big < z


# ── overnight trailing-median z ─────────────────────────────────────────────────────────────────

def test_trailing_sq_excludes_today_and_bounds_window():
    series = [float(i) for i in range(100)]
    # i = 50, window 66 → indices 0..49 (today excluded).
    w = measures.trailing_sq(series, 50, window=66)
    assert w == series[0:50]
    # i = 90, window 66 → the 66 before 90, i.e. 24..89.
    w2 = measures.trailing_sq(series, 90, window=66)
    assert w2 == series[24:90] and len(w2) == 66


def test_overnight_z_value_and_min_history():
    prev = [0.04] * 30                 # median 0.04, sqrt = 0.2
    assert math.isclose(measures.overnight_z(0.6, prev), 0.6 / 0.2)
    # Fewer than ON_MIN_HISTORY previous sessions → None.
    assert measures.overnight_z(0.6, [0.04] * (ON_MIN_HISTORY - 1)) is None
    # Exactly ON_MIN_HISTORY is enough.
    assert measures.overnight_z(0.6, [0.04] * ON_MIN_HISTORY) is not None


def test_overnight_z_zero_median_is_null():
    # A quiet name: more than half of the window is exactly 0 → median 0 → None (no divide).
    prev = [0.0] * 20 + [0.01] * 10
    assert measures.overnight_z(0.5, prev) is None


def test_overnight_flag_uses_ge_threshold():
    prev = [0.01] * 30                 # median 0.01, sqrt 0.1
    z_at = measures.overnight_z(Z_OVERNIGHT * 0.1, prev)   # exactly Z_OVERNIGHT
    assert math.isclose(z_at, Z_OVERNIGHT)
    assert measures.overnight_flag(z_at) == 1              # >= is a jump
    assert measures.overnight_flag(measures.overnight_z(0.3, prev)) == 0
    assert measures.overnight_flag(None) is None


# ── SPY adjustment ──────────────────────────────────────────────────────────────────────────────

def test_overnight_ret_adj_is_difference_and_null_rules():
    assert math.isclose(measures.overnight_ret_adj(0.05, 0.02, is_spy=False), 0.03)
    assert measures.overnight_ret_adj(0.05, 0.02, is_spy=True) is None    # SPY itself
    assert measures.overnight_ret_adj(0.05, None, is_spy=False) is None   # SPY has no row that date
    assert measures.overnight_ret_adj(None, 0.02, is_spy=False) is None


def test_constants():
    assert math.isclose(THETA, (math.pi / 2) ** 2 + math.pi - 5)
    assert math.isclose(Z_INTRADAY, 3.0902323, abs_tol=1e-6)   # Φ⁻¹(0.999)
    assert Z_OVERNIGHT == 4.0
