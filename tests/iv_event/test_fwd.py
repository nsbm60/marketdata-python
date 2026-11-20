"""The pure window math of method fwd_v1: node selection, effective days, normal/excess, earnings."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import pytest

from ml.shared.calendar import is_trading_day, next_trading_day, session_close_et

from iv_event import (
    EARNINGS_CALENDAR, EARNINGS_HISTORY, STATUS_INSUFFICIENT_WINDOWS, STATUS_OK,
)
from iv_event.daytype import gap_bucket
from iv_event.fwd import (
    ExpiryNode, build_day, calendar_reaction_session, earliest_in_window, history_reaction_session,
    select_nodes,
)


def _node(tau: int, var: float, support: float) -> ExpiryNode:
    # A distinct expiry per τ; only ordering/identity matters to select_nodes.
    return ExpiryNode(expiry=date(2026, 1, 1) + timedelta(days=tau),
                      tau=tau, fit_total_var=var, support=support)


# ── node selection: support floor + the pooled-block rule ─────────────────────────────────────────

def test_pooled_block_keeps_only_its_last_expiry_and_drops_thin_support() -> None:
    e_a = _node(1, 0.01, 100)
    e_b = _node(3, 0.02, 100)
    e_c = _node(5, 0.02, 100)   # same fit as e_b → one pooled block; keep this (its last expiry)
    e_thin = _node(6, 0.02, 30)  # support < 50 → dropped before pooling
    e_e = _node(9, 0.05, 100)
    e_tiny = _node(0, 0.001, 10)  # support < 50 → dropped

    nodes, dropped_tau0 = select_nodes([e_e, e_c, e_a, e_thin, e_b, e_tiny])  # unordered input

    assert [n.tau for n in nodes] == [1, 5, 9]           # e_b collapsed into e_c (the block's last)
    assert [n.expiry for n in nodes] == [e_a.expiry, e_c.expiry, e_e.expiry]
    assert [n.fit_total_var for n in nodes] == [0.01, 0.02, 0.05]
    assert dropped_tau0 == 0


def test_tau0_expiries_are_dropped_and_windows_start_from_the_next_node() -> None:
    # An expiry dated on/before the next session (τ=0, e.g. an unscheduled closure) is not a node: the
    # anchor already represents τ=0. Windows build from the first τ>0 node, never a zero-length window.
    e0 = _node(0, 0.001, 100)   # τ=0 — dropped
    e1, e2, e3 = _node(5, 0.05, 100), _node(10, 0.10, 100), _node(15, 0.15, 100)

    nodes, dropped_tau0 = select_nodes([e0, e1, e2, e3])
    assert dropped_tau0 == 1
    assert [n.tau for n in nodes] == [5, 10, 15]
    assert e0.expiry not in {n.expiry for n in nodes}

    windows = build_day("TEST", date(2026, 2, 1), nodes, day_weight=lambda _d: 1.0,
                        list_days=_fixed_days({e1.expiry: 5, e2.expiry: 5, e3.expiry: 5}),
                        own_earnings=_no_earnings)
    assert [w.status for w in windows] == [STATUS_OK] * 3
    assert windows[0].start_expiry is None and windows[0].end_expiry == e1.expiry  # anchor → first τ>0 node


def test_zero_effective_days_raises_naming_the_window() -> None:
    # Unreachable once τ=0 nodes are dropped, but it must never be a silent division: a window with no
    # trading day (list_days returns nothing) raises, naming the underlying, day, expiries and taus.
    e1 = _node(5, 0.05, 100)
    with pytest.raises(ValueError) as excinfo:
        build_day("ACME", date(2026, 3, 2), [e1], day_weight=lambda _d: 1.0,
                  list_days=_fixed_days({e1.expiry: 0}), own_earnings=_no_earnings)
    msg = str(excinfo.value)
    assert "ACME" in msg and "2026-03-02" in msg
    assert "anchor" in msg and str(e1.expiry) in msg
    assert "tau 0->5" in msg and "eff_days" in msg


# ── effective days with a weekend and a holiday (real calendar) ────────────────────────────────────

_WEIGHTS = {1: 1.0, 2: 1.2, 3: 1.5, 4: 2.0, 5: 3.0}


def _list_days(start: date, end: date) -> List[date]:
    from ml.shared.calendar import trading_days_in
    return trading_days_in(start, end)


def _weight(d: date) -> float:
    return _WEIGHTS[gap_bucket(d)]


def _no_earnings(_s: date, _e: date) -> Optional[Tuple[date, str]]:
    return None


def test_effective_days_fold_in_a_weekend_and_a_holiday() -> None:
    # session Fri 2021-01-08; nodes at the next three Fridays. Jan 18 2021 is MLK (holiday).
    session = date(2021, 1, 8)
    e1, e2, e3 = date(2021, 1, 15), date(2021, 1, 22), date(2021, 1, 29)
    nodes = [ExpiryNode(e1, 5, 0.05, 100), ExpiryNode(e2, 9, 0.10, 100), ExpiryNode(e3, 14, 0.15, 100)]

    windows = build_day("EWZ", session, nodes, day_weight=_weight, list_days=_list_days,
                        own_earnings=_no_earnings)
    by_end = {w.end_expiry: w for w in windows}

    # anchor→Jan15: Mon 11 (gap 3, weekend) + Tue–Fri (gap 1 ×4) = 1.5 + 4 = 5.5
    assert math.isclose(by_end[e1].eff_days, 5.5)
    # Jan15→Jan22: Tue 19 (gap 4, after the Mon-18 holiday) + Wed–Fri (gap 1 ×3) = 2.0 + 3 = 5.0
    assert math.isclose(by_end[e2].eff_days, 5.0)
    # Jan22→Jan29: Mon 25 (gap 3, weekend) + Tue–Fri (gap 1 ×4) = 1.5 + 4 = 5.5
    assert math.isclose(by_end[e3].eff_days, 5.5)


# ── normal (median), excess, implied move, excess_days, and insufficient_windows ───────────────────

def _fixed_days(n_by_end: dict[date, int]):
    def list_days(_start: date, end: date) -> List[date]:
        return [end] * n_by_end[end]   # with a constant weight, eff_days == this count
    return list_days


def test_normal_is_the_window_median_and_excess_arithmetic_incl_negative() -> None:
    e1, e2, e3, e4 = (date(2026, 2, d) for d in (2, 3, 4, 5))
    nodes = [ExpiryNode(e1, 10, 0.10, 100), ExpiryNode(e2, 20, 0.20, 100),
             ExpiryNode(e3, 30, 0.60, 100), ExpiryNode(e4, 40, 0.65, 100)]
    list_days = _fixed_days({e1: 10, e2: 10, e3: 10, e4: 10})

    windows = build_day("TEST", date(2026, 2, 1), nodes, day_weight=lambda _d: 1.0,
                        list_days=list_days, own_earnings=_no_earnings)
    assert [w.status for w in windows] == [STATUS_OK] * 4
    by_end = {w.end_expiry: w for w in windows}

    # window_vars: .10, .10, .40, .05 over eff 10 each → per-day .01, .01, .04, .005;
    # median of [.005, .01, .01, .04] = (.01 + .01)/2 = .01
    normal = 0.01
    assert all(math.isclose(w.normal_daily_var, normal) for w in windows)

    big = by_end[e3]                      # excess = .40 − .01·10 = .30
    assert math.isclose(big.excess_var, 0.30)
    assert math.isclose(big.implied_move, math.sqrt(0.30))
    assert math.isclose(big.excess_days, 30.0)

    flat = by_end[e1]                     # excess = .10 − .01·10 = 0
    assert math.isclose(flat.excess_var, 0.0)
    assert math.isclose(flat.implied_move, 0.0)

    neg = by_end[e4]                      # excess = .05 − .01·10 = −.05 (below normal)
    assert math.isclose(neg.excess_var, -0.05)
    assert math.isclose(neg.implied_move, 0.0)       # the move clamps at 0
    assert math.isclose(neg.excess_days, -5.0)       # but excess_days stays negative


def test_fewer_than_three_windows_is_one_insufficient_row() -> None:
    e1, e2 = date(2026, 2, 2), date(2026, 2, 3)
    nodes = [ExpiryNode(e1, 10, 0.10, 100), ExpiryNode(e2, 20, 0.20, 100)]
    windows = build_day("TEST", date(2026, 2, 1), nodes, day_weight=lambda _d: 1.0,
                        list_days=_fixed_days({e1: 10, e2: 10}), own_earnings=_no_earnings)
    assert len(windows) == 1
    w = windows[0]
    assert w.status == STATUS_INSUFFICIENT_WINDOWS
    assert (w.start_td, w.end_td, w.eff_days) == (0, 0, 0.0)
    assert w.window_var is None and w.excess_var is None and w.implied_move is None
    assert w.support is None and w.own_earnings_date is None and w.own_earnings_source is None


def test_no_nodes_is_insufficient() -> None:
    windows = build_day("TEST", date(2026, 2, 1), [], day_weight=lambda _d: 1.0,
                        list_days=_fixed_days({}), own_earnings=_no_earnings)
    assert len(windows) == 1 and windows[0].status == STATUS_INSUFFICIENT_WINDOWS


# ── earnings attribution ───────────────────────────────────────────────────────────────────────────

def test_history_reaction_session_prefers_the_reaction_day() -> None:
    assert history_reaction_session(date(2021, 2, 1), date(2021, 2, 2)) == date(2021, 2, 2)
    assert history_reaction_session(date(2021, 2, 1), None) == date(2021, 2, 1)


def test_calendar_reaction_session_rolls_after_the_close() -> None:
    fri = date(2021, 2, 5)
    kw = dict(session_close_et=session_close_et, is_trading_day=is_trading_day,
              next_trading_day=next_trading_day)
    # after the 16:00 close → the next session (Mon 2021-02-08)
    assert calendar_reaction_session(fri, datetime(2021, 2, 5, 16, 30), **kw) == date(2021, 2, 8)
    # before the close → the same day
    assert calendar_reaction_session(fri, datetime(2021, 2, 5, 8, 0), **kw) == fri
    # no timestamp → the same day
    assert calendar_reaction_session(fri, None, **kw) == fri
    # a non-trading day → the next session
    assert calendar_reaction_session(date(2021, 2, 6), None, **kw) == date(2021, 2, 8)


def test_earliest_in_window_is_end_inclusive_start_exclusive_tie_to_history() -> None:
    hist = (date(2021, 2, 1), EARNINGS_HISTORY)
    cal = (date(2021, 2, 10), EARNINGS_CALENDAR)

    # start is exclusive: a report ON the start day does not belong to the window
    assert earliest_in_window([hist, cal], date(2021, 2, 1), date(2021, 2, 10)) == cal
    # end is inclusive: a report ON the end day does belong
    assert earliest_in_window([hist], date(2021, 1, 31), date(2021, 2, 1)) == hist
    # nothing in range
    assert earliest_in_window([hist, cal], date(2021, 2, 11), date(2021, 2, 20)) is None
    # a tie on the same reaction session → earnings_history wins
    d = date(2021, 2, 5)
    assert earliest_in_window([(d, EARNINGS_CALENDAR), (d, EARNINGS_HISTORY)],
                              date(2021, 2, 1), date(2021, 2, 10)) == (d, EARNINGS_HISTORY)
