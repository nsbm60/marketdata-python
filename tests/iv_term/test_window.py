"""iv_term's trade window is the hour before that day's close: 15:00–16:00 ET normally, 12:00–13:00
on a NYSE early-close day."""

from __future__ import annotations

import datetime as dt

from iv_term.term import _window_hour_sql


def test_normal_month_selects_hour_15() -> None:
    # March 2027 has no early-close day → a plain `= 15`.
    sql = _window_hour_sql(dt.date(2027, 3, 1), dt.date(2027, 3, 31))
    assert sql == "15"


def test_early_close_month_selects_12_for_the_half_day_and_15_otherwise() -> None:
    # November 2026: 2026-11-27 (day after Thanksgiving) is a half day → window hour 12 there, 15 else.
    sql = _window_hour_sql(dt.date(2026, 11, 1), dt.date(2026, 11, 30))
    assert sql == "if(session_date IN (toDate('2026-11-27')), 12, 15)"


def test_december_half_day() -> None:
    sql = _window_hour_sql(dt.date(2026, 12, 1), dt.date(2026, 12, 31))
    assert sql == "if(session_date IN (toDate('2026-12-24')), 12, 15)"
