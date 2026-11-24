"""The pure vol-dataset rules: the C/J split (incl. a NULL intraday flag), the earnings-date rule
(confirmed at exactly 21 days, expected via the year-ago report, unknown), reaction-session mapping,
the event-window sums (rule vs non-rule, end_td <= H), and the target / drop rules. No ClickHouse."""

from __future__ import annotations

import math
from datetime import date, timedelta

from vol_dataset import rules


# ── C/J split ──────────────────────────────────────────────────────────────────────────────────────

def test_cj_split_intraday_jump() -> None:
    c, j, unsplit = rules.cj_split(1.0, 0.1, intraday_jump=1, jump_var=0.3,
                                   overnight_jump=0, overnight_jump_adj=0)
    assert j == 0.3 and abs(c - 0.7) < 1e-12 and unsplit is False


def test_cj_split_overnight_adj_jump_counts() -> None:
    c, j, unsplit = rules.cj_split(1.0, 0.1, intraday_jump=0, jump_var=0.0,
                                   overnight_jump=0, overnight_jump_adj=1)
    assert abs(j - 0.1) < 1e-12 and abs(c - 0.9) < 1e-12 and unsplit is False


def test_cj_split_both_parts() -> None:
    c, j, unsplit = rules.cj_split(1.0, 0.1, intraday_jump=1, jump_var=0.3,
                                   overnight_jump=1, overnight_jump_adj=0)
    assert abs(j - 0.4) < 1e-12 and abs(c - 0.6) < 1e-12 and unsplit is False


def test_cj_split_null_intraday_goes_to_continuous() -> None:
    # intraday_jump NULL -> no intraday split; the intraday variance stays in C; unsplit flagged.
    c, j, unsplit = rules.cj_split(1.0, 0.1, intraday_jump=None, jump_var=None,
                                   overnight_jump=0, overnight_jump_adj=0)
    assert j == 0.0 and c == 1.0 and unsplit is True


def test_cj_split_null_intraday_but_overnight_jumped() -> None:
    c, j, unsplit = rules.cj_split(1.0, 0.1, intraday_jump=None, jump_var=None,
                                   overnight_jump=1, overnight_jump_adj=None)
    assert abs(j - 0.1) < 1e-12 and abs(c - 0.9) < 1e-12 and unsplit is True


# ── reaction session ─────────────────────────────────────────────────────────────────────────────────

def _next(d: date) -> date:
    return d + timedelta(days=1)   # fake calendar: every day is a trading day


def test_reaction_session_mapping() -> None:
    d = date(2024, 6, 14)
    assert rules.reaction_session(d, "pre-market", _next) == d
    assert rules.reaction_session(d, "post-market", _next) == d + timedelta(days=1)
    assert rules.reaction_session(d, "unknown", _next) is None
    assert rules.reaction_session(None, "pre-market", _next) is None


# ── earnings date as known at t ────────────────────────────────────────────────────────────────────

def _between(a: date, b: date) -> int:
    return (b - a).days   # fake: every calendar day is a trading day


def test_earnings_confirmed_at_exactly_21_days() -> None:
    t = date(2024, 6, 14)
    reports = [(t + timedelta(days=21), "pre-market")]   # exactly 21 days ahead -> confirmed
    e = rules.earnings_as_of(t, reports, _next, _between)
    assert e.status == "confirmed"
    assert e.earnings_date == t + timedelta(days=21)
    assert e.report_time == "pre-market"
    assert e.reaction_session == t + timedelta(days=21)   # pre-market -> that day
    assert e.earnings_td == 21


def test_earnings_just_past_21_days_is_not_confirmed() -> None:
    t = date(2024, 6, 14)
    reports = [(t + timedelta(days=22), "pre-market")]   # 22 days ahead -> outside the window
    e = rules.earnings_as_of(t, reports, _next, _between)
    assert e.status == "unknown" and e.earnings_date is None


def test_earnings_expected_via_year_ago_report() -> None:
    t = date(2024, 6, 14)
    # A past report 300 days ago; +364 lands at t+64 -> the expected date. No confirmed report.
    reports = [(t - timedelta(days=300), "post-market")]
    e = rules.earnings_as_of(t, reports, _next, _between)
    assert e.status == "expected"
    assert e.earnings_date == t - timedelta(days=300) + timedelta(days=364)
    assert e.report_time == "post-market"
    assert e.reaction_session == e.earnings_date + timedelta(days=1)   # post-market -> next day


def test_earnings_expected_prefers_the_earliest_future_candidate() -> None:
    t = date(2024, 6, 14)
    reports = [
        (t - timedelta(days=300), "pre-market"),   # +364 -> t+64
        (t - timedelta(days=350), "post-market"),  # +364 -> t+14 (earlier) -> chosen
    ]
    e = rules.earnings_as_of(t, reports, _next, _between)
    assert e.status == "expected"
    assert e.earnings_date == t - timedelta(days=350) + timedelta(days=364)
    assert e.report_time == "post-market"


def test_earnings_unknown_when_no_future_candidate() -> None:
    t = date(2024, 6, 14)
    reports = [(t - timedelta(days=400), "pre-market")]   # +364 = t-36 <= t -> no future candidate
    e = rules.earnings_as_of(t, reports, _next, _between)
    assert e.status == "unknown" and e.earnings_date is None and e.report_time == "unknown"
    assert e.reaction_session is None and e.earnings_td is None


def test_earnings_confirmed_beats_expected() -> None:
    t = date(2024, 6, 14)
    reports = [
        (t + timedelta(days=5), "post-market"),     # confirmed (within 21 ahead)
        (t - timedelta(days=350), "pre-market"),    # would be expected, but confirmed wins
    ]
    e = rules.earnings_as_of(t, reports, _next, _between)
    assert e.status == "confirmed" and e.earnings_date == t + timedelta(days=5)


# ── event-window sums ────────────────────────────────────────────────────────────────────────────────

def test_event_window_sums_rule_and_non_rule_and_horizon() -> None:
    windows = [
        (5, 3.0, 2, 3.0),     # rule: eff<=5, start<=5, excess>=2
        (8, 4.0, 1, 2.5),     # within H=10, rule (eff 4, start 1, excess 2.5)
        (5, 8.0, 2, 1.5),     # non-rule (eff 8 > 5); positive excess 1.5 -> other
        (5, 3.0, 6, 4.0),     # non-rule (start_td 6 > 5); excess 4.0 -> other
        (5, 3.0, 2, -1.0),    # non-rule (excess < 2) and negative -> contributes nothing
        (30, 3.0, 2, 9.0),    # end_td 30 > H -> excluded entirely
    ]
    rule_n, rule_exc, other_exc = rules.event_window_sums(windows, horizon=10)
    assert rule_n == 2
    assert abs(rule_exc - (3.0 + 2.5)) < 1e-12
    assert abs(other_exc - (1.5 + 4.0)) < 1e-12


# ── targets ─────────────────────────────────────────────────────────────────────────────────────────

def test_compute_targets_values() -> None:
    td, tu, reason = rules.compute_targets(0.5, 0.3, 1.0)
    assert reason is None
    assert abs(td - math.log(1.0)) < 1e-12        # 2*0.5/1.0 = 1 -> log 0
    assert abs(tu - math.log(0.6)) < 1e-12


def test_compute_targets_drops_zero_down_then_up() -> None:
    assert rules.compute_targets(0.0, 0.3, 1.0) == (None, None, "zero_realized_down")
    assert rules.compute_targets(0.5, 0.0, 1.0) == (None, None, "zero_realized_up")
