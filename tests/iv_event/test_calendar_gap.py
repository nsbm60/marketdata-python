"""The shared-calendar additions iv_event relies on: previous/next trading day and trading_days_in,
over a weekend and a Monday holiday (MLK 2021-01-18)."""

from __future__ import annotations

from datetime import date

from ml.shared.calendar import next_trading_day, previous_trading_day, trading_days_in


def test_previous_trading_day_skips_weekends_and_holidays() -> None:
    assert previous_trading_day(date(2021, 1, 11)) == date(2021, 1, 8)   # Mon → the Friday before
    assert previous_trading_day(date(2021, 1, 19)) == date(2021, 1, 15)  # Tue after MLK → Fri 15
    assert previous_trading_day(date(2021, 1, 12)) == date(2021, 1, 11)  # an ordinary Tuesday


def test_next_trading_day_skips_weekends_and_holidays() -> None:
    assert next_trading_day(date(2021, 1, 8)) == date(2021, 1, 11)    # Fri → the Monday after
    assert next_trading_day(date(2021, 1, 15)) == date(2021, 1, 19)   # Fri before MLK → Tue 19
    assert next_trading_day(date(2021, 1, 6)) == date(2021, 1, 7)


def test_trading_days_in_is_exclusive_start_inclusive_end() -> None:
    # (Jan 8, Jan 15] 2021: Mon 11, Tue 12, Wed 13, Thu 14, Fri 15
    assert trading_days_in(date(2021, 1, 8), date(2021, 1, 15)) == [
        date(2021, 1, 11), date(2021, 1, 12), date(2021, 1, 13), date(2021, 1, 14), date(2021, 1, 15)]
    # (Jan 15, Jan 22] 2021 drops Mon 18 (MLK): Tue 19, Wed 20, Thu 21, Fri 22
    assert trading_days_in(date(2021, 1, 15), date(2021, 1, 22)) == [
        date(2021, 1, 19), date(2021, 1, 20), date(2021, 1, 21), date(2021, 1, 22)]
    # empty / reversed ranges
    assert trading_days_in(date(2021, 6, 15), date(2021, 6, 15)) == []
    assert trading_days_in(date(2021, 6, 20), date(2021, 6, 15)) == []
