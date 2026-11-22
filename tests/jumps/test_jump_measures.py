"""Offline tests for the jump_daily measures reference (jumps/measures.py) — no ClickHouse."""

import datetime as dt
import math

from jumps import (
    BETA_MAX_AGE_DAYS, BETA_MIN_HISTORY, MED_CHI2_1, ON_MAX_AGE_DAYS, ON_MIN_HISTORY, THETA,
    Z_INTRADAY, Z_OVERNIGHT, _JUMP_LOOKBACK_DAYS, measures,
)


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

def test_recent_sq_excludes_today_and_bounds_window():
    base = dt.date(2024, 6, 1)
    dated = [(base + dt.timedelta(days=i), float(i)) for i in range(100)]  # dense, 1 day apart
    today = dated[80][0]
    r = measures.recent_sq(dated, today, window=66, max_age=10_000)  # age slack
    # excludes today (index 80); the most recent 66 before it are indices 14..79.
    assert r == [float(i) for i in range(14, 80)] and len(r) == 66


def test_recent_sq_age_limit_drops_sessions_older_than_the_gap():
    today = dt.date(2020, 1, 20)
    old = [(dt.date(2018, 6, 1) + dt.timedelta(days=i), 0.09) for i in range(40)]      # >200 days old
    recent = [(today - dt.timedelta(days=12 - i), 0.04) for i in range(10)]            # 10 within ~2 weeks
    r = measures.recent_sq(old + recent, today, window=66, max_age=ON_MAX_AGE_DAYS)
    # The 2018 sessions are all more than 200 days old → dropped; only the 10 recent remain.
    assert r == [0.04] * 10


def test_overnight_z_null_after_gap_with_too_few_recent_sessions():
    # The SMCI case: 66 old sessions (2018) then only a handful of recent ones. The ROWS-66 frame is
    # full of old sessions, but the age limit drops them, leaving fewer than ON_MIN_HISTORY → None.
    today = dt.date(2020, 1, 20)
    old = [(dt.date(2018, 6, 1) + dt.timedelta(days=i), 0.09) for i in range(66)]
    recent = [(today - dt.timedelta(days=5 - i), 0.04) for i in range(5)]              # 5 < ON_MIN_HISTORY
    r = measures.recent_sq(old + recent, today, window=66, max_age=ON_MAX_AGE_DAYS)
    assert len(r) == 5
    assert measures.overnight_z(0.3, r) is None


def test_overnight_z_value_and_min_history():
    prev = [0.04] * 30                 # median 0.04
    expected = 0.6 / math.sqrt(0.04 / MED_CHI2_1)   # scaled to a σ multiple
    assert math.isclose(measures.overnight_z(0.6, prev), expected)
    # Fewer than ON_MIN_HISTORY previous sessions → None.
    assert measures.overnight_z(0.6, [0.04] * (ON_MIN_HISTORY - 1)) is None
    # Exactly ON_MIN_HISTORY is enough.
    assert measures.overnight_z(0.6, [0.04] * ON_MIN_HISTORY) is not None


def test_overnight_z_scale_makes_normal_returns_sit_near_one_sigma():
    # If o ~ N(0, σ²), the median of o² is MED_CHI2_1·σ²; dividing by MED_CHI2_1 recovers σ², so a
    # 1σ move scores ~1.0 (the pre-fix scale would score ~1/sqrt(MED_CHI2_1) ≈ 1.48).
    sigma = 0.02
    prev = [MED_CHI2_1 * sigma * sigma] * 40   # a window whose median is exactly MED_CHI2_1·σ²
    assert math.isclose(measures.overnight_z(sigma, prev), 1.0)


def test_overnight_z_zero_median_is_null():
    # A quiet name: more than half of the window is exactly 0 → median 0 → None (no divide).
    prev = [0.0] * 20 + [0.01] * 10
    assert measures.overnight_z(0.5, prev) is None


def test_overnight_flag_uses_ge_threshold():
    prev = [0.01] * 30
    ret_at = Z_OVERNIGHT * math.sqrt(0.01 / MED_CHI2_1)   # scores exactly Z_OVERNIGHT
    z_at = measures.overnight_z(ret_at, prev)
    assert math.isclose(z_at, Z_OVERNIGHT)
    assert measures.overnight_flag(z_at) == 1                           # >= is a jump
    assert measures.overnight_flag(measures.overnight_z(ret_at * 0.5, prev)) == 0
    assert measures.overnight_flag(None) is None


# ── beta market adjustment ────────────────────────────────────────────────────────────────────────

def test_beta_recovers_known_slope():
    base, today = dt.date(2023, 1, 1), dt.date(2024, 1, 1)
    # o = 2·o_spy + 0.5 exactly → OLS slope 2.0 (intercept irrelevant to the slope).
    pairs = [(base + dt.timedelta(days=3 * i), 2.0 * (0.01 * math.sin(i)) + 0.5, 0.01 * math.sin(i))
             for i in range(80)]
    assert math.isclose(measures.beta(pairs, today), 2.0, abs_tol=1e-9)


def test_beta_null_below_min_history():
    base, today = dt.date(2023, 6, 1), dt.date(2024, 1, 1)
    pairs = [(base + dt.timedelta(days=i), 0.5 * (0.01 * math.cos(i)), 0.01 * math.cos(i))
             for i in range(BETA_MIN_HISTORY - 1)]
    assert measures.beta(pairs, today) is None


def test_beta_null_when_spy_variance_zero():
    base, today = dt.date(2023, 6, 1), dt.date(2024, 1, 1)
    pairs = [(base + dt.timedelta(days=i), 0.01 * i, 0.0) for i in range(70)]  # o_spy flat → var 0
    assert measures.beta(pairs, today) is None


def test_beta_excludes_sessions_where_spy_absent():
    base, today = dt.date(2023, 1, 1), dt.date(2024, 1, 1)
    good = [(base + dt.timedelta(days=3 * i), 2.0 * (0.01 * math.sin(i)) + 0.1, 0.01 * math.sin(i))
            for i in range(70)]
    missing = [(base + dt.timedelta(days=3 * i + 1), 0.9, None) for i in range(70)]  # SPY absent → dropped
    pairs = sorted(good + missing, key=lambda t: t[0])
    assert math.isclose(measures.beta(pairs, today), 2.0, abs_tol=1e-9)


def test_beta_age_limit_drops_old_sessions():
    today = dt.date(2024, 1, 1)
    old = [(dt.date(2020, 1, 1) + dt.timedelta(days=i), 3.0 * (0.01 * math.sin(i)), 0.01 * math.sin(i))
           for i in range(70)]                                              # >400 days old → dropped
    recent = [(today - dt.timedelta(days=5 * (70 - i)), 2.0 * (0.01 * math.cos(i)), 0.01 * math.cos(i))
              for i in range(70)]                                           # within ~345 days → kept
    assert math.isclose(measures.beta(old + recent, today), 2.0, abs_tol=1e-9)


# ── overnight adjustment and null rules ─────────────────────────────────────────────────────────

def test_overnight_ret_adj_uses_beta_and_null_rules():
    assert math.isclose(measures.overnight_ret_adj(0.05, 0.02, beta_hat=1.5, is_spy=False), 0.05 - 1.5 * 0.02)
    assert measures.overnight_ret_adj(0.05, 0.02, beta_hat=1.5, is_spy=True) is None    # SPY itself
    assert measures.overnight_ret_adj(0.05, 0.02, beta_hat=None, is_spy=False) is None  # β not estimable
    assert measures.overnight_ret_adj(0.05, None, beta_hat=1.5, is_spy=False) is None   # SPY absent today
    assert measures.overnight_ret_adj(None, 0.02, beta_hat=1.5, is_spy=False) is None


def test_constants():
    assert math.isclose(THETA, (math.pi / 2) ** 2 + math.pi - 5)
    assert math.isclose(Z_INTRADAY, 3.0902323, abs_tol=1e-6)   # Φ⁻¹(0.999)
    assert Z_OVERNIGHT == 4.0
    assert math.isclose(MED_CHI2_1, 0.4549364, abs_tol=1e-6)   # median of χ²₁


# ── read lookback must be the SUM of the two age limits ─────────────────────────────────────────────

def _z_adj_from_series(series, target_date):
    """Compute one row's overnight_z_adj end to end from a (date, o, o_spy) series: the row's own
    beta → its overnight_ret_adj, then the trailing median of the previous days' overnight_ret_adj²
    (each previous day's o_adj uses that day's own beta). Mirrors the SQL's dependency chain."""
    D, o_D, spy_D = next(t for t in series if t[0] == target_date)
    o_adj_D = measures.overnight_ret_adj(o_D, spy_D, measures.beta(series, D), is_spy=False)
    dated_sqs = []
    for (d, o, s) in series:
        if d >= D:
            continue
        oa = measures.overnight_ret_adj(o, s, measures.beta(series, d), is_spy=False)
        if oa is not None:
            dated_sqs.append((d, oa * oa))
    recent = measures.recent_sq(dated_sqs, D)   # defaults: ON_WINDOW rows, ON_MAX_AGE_DAYS age
    return measures.overnight_z(o_adj_D, recent)


def test_adjusted_z_lookback_is_the_sum_of_age_limits():
    # The constant must be the sum, not the larger.
    assert _JUMP_LOOKBACK_DAYS == ON_MAX_AGE_DAYS + BETA_MAX_AGE_DAYS

    # A series spaced so both age caps bind: the median reaches ~ON_MAX_AGE_DAYS back, and the oldest
    # median day's beta reaches ~BETA_MAX_AGE_DAYS before it — so the row depends on data up to the
    # sum (600 days) back. o = 1.2·o_spy + idiosyncratic, so beta depends on the full window.
    series = []
    d0 = dt.date(2018, 1, 1)
    for i in range(300):                       # 4-day spacing → ~1196-day span
        d = d0 + dt.timedelta(days=4 * i)
        s = 0.01 * math.sin(0.3 * i)
        o = 1.2 * s + 0.004 * math.cos(0.5 * i)
        series.append((d, o, s))
    D = series[-1][0]

    z_full = _z_adj_from_series(series, D)
    assert z_full is not None

    lb = ON_MAX_AGE_DAYS + BETA_MAX_AGE_DAYS
    trim_sum = [t for t in series if t[0] >= D - dt.timedelta(days=lb)]
    assert _z_adj_from_series(trim_sum, D) == z_full      # the sum reads everything the row uses

    # Trimming at only the larger age limit (BETA_MAX_AGE_DAYS) truncates the oldest median days'
    # betas, so the adjusted z silently changes — this is the bug the sum fixes.
    trim_short = [t for t in series if t[0] >= D - dt.timedelta(days=BETA_MAX_AGE_DAYS)]
    assert _z_adj_from_series(trim_short, D) != z_full
