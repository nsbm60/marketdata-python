"""The pure outcome math of method out_v1: window-day selection, realized/jump stats, release with
TV(E,E)=0 and missing curve days, daytype-weight use, crush ties, the corporate-action window, the
explained precedence, and the NULL rules. No ClickHouse — everything is a plain dict or a callable."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional

from iv_event import outcome as oc
from iv_event.outcome import JumpRow, WindowSpec


# ── fixtures / tiny calendar ───────────────────────────────────────────────────────────────────────

def _list_days(start_exclusive: date, end_inclusive: date) -> List[date]:
    """Every calendar day in (start, end] — a dense calendar is enough for the pure math tests."""
    out: List[date] = []
    d = start_exclusive + timedelta(days=1)
    while d <= end_inclusive:
        out.append(d)
        d += timedelta(days=1)
    return out


def _prev(d: date) -> date:
    return d - timedelta(days=1)


def _spec(start_expiry: Optional[date], end_expiry: date, *, eff_days: float = 3.0,
          normal: float = 0.001, own_earn: Optional[date] = None) -> WindowSpec:
    return WindowSpec(
        underlying="TEST", session_date=date(2024, 1, 1), start_expiry=start_expiry,
        end_expiry=end_expiry, end_td=5, eff_days=eff_days, normal_daily_var=normal,
        own_earnings_date=own_earn)


# ── window-day selection: anchor and non-anchor ────────────────────────────────────────────────────

def test_window_days_anchor_uses_session_date() -> None:
    # anchor (S None): s < d <= E
    days = oc.window_days(date(2024, 1, 1), None, date(2024, 1, 4), _list_days)
    assert days == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]


def test_window_days_non_anchor_uses_max_of_session_and_start_expiry() -> None:
    # non-anchor: max(s, S) < d <= E. S is after s, so it is the lower bound.
    days = oc.window_days(date(2024, 1, 1), date(2024, 1, 3), date(2024, 1, 6), _list_days)
    assert days == [date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 6)]


# ── realized: the all-or-nothing NULL rule ─────────────────────────────────────────────────────────

def test_realized_var_sums_only_when_every_window_day_has_a_row() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    full = {d: 0.01 for d in days}
    n, rv = oc.realized(days, full)
    assert n == 3 and abs(rv - 0.03) < 1e-12

    partial = {days[0]: 0.01, days[1]: 0.02}   # one day missing → no partial sum
    n, rv = oc.realized(days, partial)
    assert n == 2 and rv is None


def test_realized_empty_window_is_null_not_zero() -> None:
    n, rv = oc.realized([], {})
    assert n == 0 and rv is None


# ── jump stats ─────────────────────────────────────────────────────────────────────────────────────

def test_jump_stats_counts_any_flag_and_tracks_first_and_max_abs_adj() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    rows: Dict[date, JumpRow] = {
        days[0]: JumpRow(overnight_jump=0, overnight_jump_adj=0, intraday_jump=0, overnight_ret_adj=-0.01),
        days[1]: JumpRow(overnight_jump=1, overnight_jump_adj=1, intraday_jump=0, overnight_ret_adj=0.05),  # jump (overnight)
        days[2]: JumpRow(overnight_jump=0, overnight_jump_adj=0, intraday_jump=1, overnight_ret_adj=0.02),  # jump (intraday)
        # days[3] has no jump_daily row at all
    }
    n, first, max_abs = oc.jump_stats(days, rows)
    assert n == 2
    assert first == days[1]
    assert abs(max_abs - 0.05) < 1e-12


def test_jump_stats_counts_raw_overnight_jump_when_adj_is_null() -> None:
    # The SPY case: overnight_jump = 1 but overnight_jump_adj is NULL (benchmark → no adjusted flag)
    # and intraday_jump = 0. The raw overnight flag alone must make it a jump day and set first_jump.
    days = [date(2024, 8, 2), date(2024, 8, 5), date(2024, 8, 6)]
    rows = {
        days[0]: JumpRow(overnight_jump=0, overnight_jump_adj=None, intraday_jump=0, overnight_ret_adj=None),
        days[1]: JumpRow(overnight_jump=1, overnight_jump_adj=None, intraday_jump=0, overnight_ret_adj=None),
        days[2]: JumpRow(overnight_jump=0, overnight_jump_adj=None, intraday_jump=0, overnight_ret_adj=None),
    }
    n, first, max_abs = oc.jump_stats(days, rows)
    assert n == 1
    assert first == days[1]
    assert max_abs is None   # adj is NULL throughout (SPY) → max_abs_ret_adj stays NULL


def test_jump_stats_max_abs_ret_adj_null_when_all_null() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    rows = {d: JumpRow(overnight_jump=0, overnight_jump_adj=0, intraday_jump=0, overnight_ret_adj=None)
            for d in days}
    n, first, max_abs = oc.jump_stats(days, rows)
    assert n == 0 and first is None and max_abs is None


# ── releases: TV(E,E)=0, missing curve days, daytype weight, ties ──────────────────────────────────

def test_release_uses_tv_e_e_zero_and_the_daytype_weight() -> None:
    E = date(2024, 1, 5)
    days = [date(2024, 1, 4), E]   # last day is E itself → TV(E, E) = 0
    # TV on the curve: d3 -> 0.10, d4 -> 0.06. The release on E uses TV(E_prev=d4) - TV(E)=0.
    curve = {(date(2024, 1, 3), E): 0.10, (date(2024, 1, 4), E): 0.06}

    def curve_tv(session: date, expiry: date) -> Optional[float]:
        return curve.get((session, expiry))

    # normal_daily_var 0.001, weight 2.0 → excess = release - 0.001*2.0 = release - 0.002.
    n_rel, crush_date, crush_xs = oc.releases(
        days, E, normal_daily_var=0.001, prev_trading_day=_prev, curve_tv=curve_tv,
        day_weight=lambda _d: 2.0)
    # day 2024-01-04: release = TV(01-03) - TV(01-04) = 0.10 - 0.06 = 0.04 → excess 0.038
    # day E (01-05): release = TV(01-04) - TV(E,E)=0 = 0.06 → excess 0.058  ← larger → crush
    assert n_rel == 2
    assert crush_date == E
    assert abs(crush_xs - (0.06 - 0.002)) < 1e-12


def test_release_skips_days_with_a_missing_curve_value() -> None:
    E = date(2024, 1, 5)
    days = [date(2024, 1, 3), date(2024, 1, 4)]
    # 01-03 needs TV(01-02) and TV(01-03); TV(01-02) is absent → no release that day.
    curve = {(date(2024, 1, 3), E): 0.10, (date(2024, 1, 4), E): 0.06}

    def curve_tv(session: date, expiry: date) -> Optional[float]:
        return curve.get((session, expiry))

    n_rel, crush_date, crush_xs = oc.releases(
        days, E, normal_daily_var=0.0, prev_trading_day=_prev, curve_tv=curve_tv,
        day_weight=lambda _d: 1.0)
    assert n_rel == 1                 # only 01-04 has both TV(01-03) and TV(01-04)
    assert crush_date == date(2024, 1, 4)
    assert abs(crush_xs - 0.04) < 1e-12


def test_release_no_day_has_a_release_gives_nulls() -> None:
    E = date(2024, 1, 5)
    days = [date(2024, 1, 3), date(2024, 1, 4)]

    def curve_tv(_s: date, _e: date) -> Optional[float]:
        return None   # nothing on the curve → no releases anywhere

    n_rel, crush_date, crush_xs = oc.releases(
        days, E, normal_daily_var=0.0, prev_trading_day=_prev, curve_tv=curve_tv,
        day_weight=lambda _d: 1.0)
    assert n_rel == 0 and crush_date is None and crush_xs is None


def test_crush_tie_takes_the_earliest_day() -> None:
    E = date(2024, 1, 9)
    d1, d2 = date(2024, 1, 4), date(2024, 1, 6)
    days = [d1, d2]
    # Both days have the same release 0.05 (a tie). Earliest (d1) must win.
    curve = {
        (_prev(d1), E): 0.20, (d1, E): 0.15,   # release 0.05
        (_prev(d2), E): 0.10, (d2, E): 0.05,   # release 0.05
    }

    def curve_tv(session: date, expiry: date) -> Optional[float]:
        return curve.get((session, expiry))

    n_rel, crush_date, crush_xs = oc.releases(
        days, E, normal_daily_var=0.0, prev_trading_day=_prev, curve_tv=curve_tv,
        day_weight=lambda _d: 1.0)
    assert n_rel == 2 and crush_date == d1 and abs(crush_xs - 0.05) < 1e-12


# ── corporate action window [s, E] ─────────────────────────────────────────────────────────────────

def test_corporate_action_in_window_is_inclusive_on_both_ends() -> None:
    s, E = date(2024, 7, 24), date(2024, 9, 20)
    assert oc.corporate_action_in_window([date(2024, 9, 20)], [], s, E) == 1   # E inclusive
    assert oc.corporate_action_in_window([date(2024, 7, 24)], [], s, E) == 1   # s inclusive
    assert oc.corporate_action_in_window([date(2024, 9, 21)], [], s, E) == 0   # just past E
    assert oc.corporate_action_in_window([], [date(2024, 8, 1)], s, E) == 1    # a split counts too
    assert oc.corporate_action_in_window([date(2024, 7, 23)], [date(2023, 1, 1)], s, E) == 0


# ── explained precedence ───────────────────────────────────────────────────────────────────────────

def test_explained_precedence_own_earnings_then_market_wide_then_none() -> None:
    mw = {date(2024, 8, 5)}

    def is_mw(d: date) -> bool:
        return d in mw

    # own_earnings always wins, even if the jump/crush day is market-wide.
    assert oc.explained(date(2024, 8, 5), date(2024, 8, 5), None, is_mw) == "own_earnings"
    # no own earnings, but first_jump_date is market-wide → market_wide
    assert oc.explained(None, date(2024, 8, 5), None, is_mw) == "market_wide"
    # no own earnings, crush_date is market-wide (first_jump None) → market_wide
    assert oc.explained(None, None, date(2024, 8, 5), is_mw) == "market_wide"
    # nothing market-wide → none
    assert oc.explained(None, date(2024, 1, 2), date(2024, 1, 3), is_mw) == "none"


# ── build_outcome: realized_excess_var NULL rule and assembly ──────────────────────────────────────

def test_build_outcome_realized_excess_var_null_when_realized_var_null() -> None:
    E = date(2024, 1, 4)
    spec = _spec(None, E, eff_days=3.0, normal=0.001)
    days = oc.window_days(spec.session_date, None, E, _list_days)   # 3 days
    rv = {days[0]: 0.01}   # incomplete → realized_var NULL → realized_excess_var NULL

    _days, out = oc.build_outcome(
        spec, list_days=_list_days, prev_trading_day=_prev, day_weight=lambda _d: 1.0,
        rv_by_date=rv, jump_by_date={}, curve_tv=lambda _s, _e: None, ca_dates=(), split_dates=(),
        is_market_wide=lambda _d: False)
    assert out.n_days == 3 and out.n_rv_days == 1
    assert out.realized_var is None and out.realized_excess_var is None


def test_build_outcome_realized_excess_var_subtracts_normal_times_eff_days() -> None:
    E = date(2024, 1, 4)
    spec = _spec(None, E, eff_days=3.0, normal=0.001)
    days = oc.window_days(spec.session_date, None, E, _list_days)
    rv = {d: 0.01 for d in days}   # complete → realized_var = 0.03

    _days, out = oc.build_outcome(
        spec, list_days=_list_days, prev_trading_day=_prev, day_weight=lambda _d: 1.0,
        rv_by_date=rv, jump_by_date={}, curve_tv=lambda _s, _e: None, ca_dates=(), split_dates=(),
        is_market_wide=lambda _d: False)
    assert abs(out.realized_var - 0.03) < 1e-12
    assert abs(out.realized_excess_var - (0.03 - 0.001 * 3.0)) < 1e-12
