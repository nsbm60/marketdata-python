"""
The pure math of the vol dataset — no ClickHouse, no pandas, no calendar of its own (calendar helpers
are passed in, so the arithmetic is deterministic and unit-tested in isolation; mirrors iv_event/fwd.py
and iv_event/outcome.py):

  - cj_split: the day's continuous / jump split (and whether the intraday flag was NULL);
  - earnings_as_of: the earnings date as KNOWN at t — a confirmed report published within 21 days
    ahead, else a past report's date + 364 as the expected date, else unknown — with its reaction
    session and the trading days from t to it;
  - event_window_sums: the priced-event counts/sums over a session's windows with end_td <= H;
  - compute_targets: the log semivariance-ratio targets, or the drop reason for a zero realized sum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Callable, List, Optional, Sequence, Tuple

from ml.shared.calendar import next_trading_day as _next_trading_day
from ml.shared.calendar import trading_days_between as _trading_days_between

from . import (
    EARNINGS_CONFIRMED_DAYS, EARNINGS_STATUS_CONFIRMED, EARNINGS_STATUS_EXPECTED,
    EARNINGS_STATUS_UNKNOWN, EARNINGS_YEAR_DAYS, EVENT_MAX_EFF_DAYS, EVENT_MAX_START_TD,
    EVENT_MIN_EXCESS_DAYS, REPORT_TIME_POST, REPORT_TIME_PRE, REPORT_TIME_UNKNOWN,
)


# ── C/J split ─────────────────────────────────────────────────────────────────────────────────────

def cj_split(rv_total: float, rv_overnight: float, intraday_jump: Optional[int],
             jump_var: Optional[float], overnight_jump: Optional[int],
             overnight_jump_adj: Optional[int]) -> Tuple[float, float, bool]:
    """(continuous, jump, unsplit) for one day. The jump part is the flagged intraday jump variance
    (jump_var when intraday_jump = 1) plus the overnight variance when the overnight move jumped
    (overnight_jump = 1 OR overnight_jump_adj = 1); the continuous part is the rest of rv_total. A day
    whose intraday_jump is NULL has NO intraday split — its intraday variance stays in the continuous
    part (unsplit = True) — so this is visible via n_unsplit_days, not hidden in the jump part."""
    intraday_part = float(jump_var) if (intraday_jump == 1 and jump_var is not None) else 0.0
    overnight_part = float(rv_overnight) if (overnight_jump == 1 or overnight_jump_adj == 1) else 0.0
    jump = intraday_part + overnight_part
    return rv_total - jump, jump, intraday_jump is None


# ── the earnings date as known at t ─────────────────────────────────────────────────────────────────

def _report_time(vendor_report_time: Optional[str]) -> str:
    """The vendor's pre/post-market tag, or 'unknown' for a blank/unrecognised time."""
    if vendor_report_time == REPORT_TIME_PRE:
        return REPORT_TIME_PRE
    if vendor_report_time == REPORT_TIME_POST:
        return REPORT_TIME_POST
    return REPORT_TIME_UNKNOWN


def reaction_session(earnings_date: Optional[date], report_time: str,
                     next_trading_day: Callable[[date], date] = _next_trading_day) -> Optional[date]:
    """The session that reacts to a report: pre-market -> that day; post-market -> the next trading
    day; unknown time -> NULL (no datable reaction)."""
    if earnings_date is None:
        return None
    if report_time == REPORT_TIME_PRE:
        return earnings_date
    if report_time == REPORT_TIME_POST:
        return next_trading_day(earnings_date)
    return None


@dataclass(frozen=True)
class EarningsAsOf:
    """The earnings view at t: the date and vendor time chosen, its status, the reaction session, and
    the trading days from t to that session (None when there is no datable session after t)."""

    earnings_date: Optional[date]
    report_time: str
    status: str
    reaction_session: Optional[date]
    earnings_td: Optional[int]


def earnings_as_of(t: date, reports: Sequence[Tuple[date, Optional[str]]],
                   next_trading_day: Callable[[date], date] = _next_trading_day,
                   trading_days_between: Callable[[date, date], int] = _trading_days_between
                   ) -> EarningsAsOf:
    """The earnings date as known at t, from the name's earnings_history reports (reported_date, time):
      - confirmed: the earliest report with reported_date within EARNINGS_CONFIRMED_DAYS ahead of t
        (reported_date - 21 <= t) whose reaction session is after t — the vendor publishes a confirmed
        date ahead, so this is known at t;
      - else expected: over the reports already past at t (reported_date <= t), each reported_date +
        EARNINGS_YEAR_DAYS; the earliest of those after t, with that report's time — never looks at the
        upcoming report, so it is look-ahead-safe;
      - else unknown.
    Then the reaction session and the trading days from t to it."""
    chosen_date: Optional[date] = None
    chosen_time = REPORT_TIME_UNKNOWN
    status = EARNINGS_STATUS_UNKNOWN

    # confirmed — a report published within the confirmation window ahead, reacting after t.
    confirmed: List[Tuple[date, str]] = []
    for reported_date, vendor_time in reports:
        rt = _report_time(vendor_time)
        if (reported_date - t).days > EARNINGS_CONFIRMED_DAYS:
            continue
        rs = reaction_session(reported_date, rt, next_trading_day)
        effective = rs if rs is not None else reported_date
        if effective > t:
            confirmed.append((reported_date, rt))
    if confirmed:
        chosen_date, chosen_time = min(confirmed, key=lambda r: r[0])
        status = EARNINGS_STATUS_CONFIRMED
    else:
        # expected — the earliest year-ago-plus-364 date after t, from reports already past at t.
        expected: List[Tuple[date, str]] = []
        for reported_date, vendor_time in reports:
            if reported_date > t:
                continue
            cand = reported_date + _days(EARNINGS_YEAR_DAYS)
            if cand > t:
                expected.append((cand, _report_time(vendor_time)))
        if expected:
            chosen_date, chosen_time = min(expected, key=lambda r: r[0])
            status = EARNINGS_STATUS_EXPECTED

    rs = reaction_session(chosen_date, chosen_time, next_trading_day)
    td = trading_days_between(t, rs) if (rs is not None and rs > t) else None
    return EarningsAsOf(chosen_date, chosen_time, status, rs, td)


def _days(n: int):
    from datetime import timedelta
    return timedelta(days=n)


# ── priced-event sums over a session's windows ────────────────────────────────────────────────────

def event_window_sums(windows: Sequence[Tuple[int, float, int, float]], horizon: int
                      ) -> Tuple[int, float, float]:
    """(rule_count, rule_excess_days, other_positive_excess_days) over the session's windows with
    end_td <= horizon. A window meets the detector's event rule when eff_days <= EVENT_MAX_EFF_DAYS,
    start_td <= EVENT_MAX_START_TD and excess_days >= EVENT_MIN_EXCESS_DAYS. The non-rule windows
    contribute only their positive excess_days. Each window is (end_td, eff_days, start_td,
    excess_days)."""
    rule_count = 0
    rule_excess = 0.0
    other_excess = 0.0
    for end_td, eff_days, start_td, excess_days in windows:
        if end_td > horizon:
            continue
        if (eff_days <= EVENT_MAX_EFF_DAYS and start_td <= EVENT_MAX_START_TD
                and excess_days >= EVENT_MIN_EXCESS_DAYS):
            rule_count += 1
            rule_excess += excess_days
        elif excess_days > 0:
            other_excess += excess_days
    return rule_count, rule_excess, other_excess


# ── targets ───────────────────────────────────────────────────────────────────────────────────────

def compute_targets(realized_down_var: float, realized_up_var: float, implied_total_var: float
                    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """(target_down, target_up, drop_reason). target = log(2 * realized semivariance / implied total
    variance). A zero realized sum makes the log undefined — the row is dropped with the reason
    'zero_realized_down' or 'zero_realized_up' (down checked first). implied_total_var is the ok
    horizon row's fit_total_var (> 0)."""
    if realized_down_var <= 0.0:
        return None, None, "zero_realized_down"
    if realized_up_var <= 0.0:
        return None, None, "zero_realized_up"
    target_down = math.log(2.0 * realized_down_var / implied_total_var)
    target_up = math.log(2.0 * realized_up_var / implied_total_var)
    return target_down, target_up, None
