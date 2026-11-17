"""The one shared trading calendar (ml.shared.calendar): holidays, observed rules, special closures,
the Juneteenth ≥2022 gate, and trading_days_between."""

from __future__ import annotations

from datetime import date

from ml.shared.calendar import is_trading_day, trading_days_between


def test_juneteenth_is_a_holiday_only_from_2022() -> None:
    # 2016–2021: NYSE traded on June 19 (observed) — the data starts in 2016.
    assert is_trading_day(date(2020, 6, 19))   # Fri
    assert is_trading_day(date(2021, 6, 18))   # Jun 19 2021 Sat → observed Fri 18, but not yet a holiday
    # 2022 onward: a market holiday (observed).
    assert not is_trading_day(date(2022, 6, 20))  # Jun 19 2022 Sun → observed Mon 20
    assert not is_trading_day(date(2023, 6, 19))  # Mon


def test_good_friday_is_a_holiday() -> None:
    assert not is_trading_day(date(2021, 4, 2))   # Good Friday 2021 (Easter Apr 4)
    assert not is_trading_day(date(2026, 4, 3))   # Good Friday 2026 (Easter Apr 5)
    assert is_trading_day(date(2021, 4, 1))       # the Thursday before trades


def test_observed_holiday_rules() -> None:
    # New Year's on a Saturday is NOT observed on the Friday before: NYSE traded 2021-12-31.
    assert is_trading_day(date(2021, 12, 31))
    # July 4 2020 Sat → observed Friday 2020-07-03 (closed).
    assert not is_trading_day(date(2020, 7, 3))
    # July 4 2021 Sun → observed Monday 2021-07-05 (closed).
    assert not is_trading_day(date(2021, 7, 5))
    # Christmas 2021 Sat → observed Friday 2021-12-24 (closed).
    assert not is_trading_day(date(2021, 12, 24))
    # A Sunday New Year (2023-01-01) → observed Monday 2023-01-02 (closed).
    assert not is_trading_day(date(2023, 1, 2))


def test_special_closures() -> None:
    assert not is_trading_day(date(2018, 12, 5))   # George H.W. Bush
    assert not is_trading_day(date(2025, 1, 9))    # Jimmy Carter
    assert is_trading_day(date(2018, 12, 6))       # the day after trades


def test_weekends_are_not_trading_days() -> None:
    assert not is_trading_day(date(2026, 1, 17))   # Saturday
    assert not is_trading_day(date(2026, 1, 18))   # Sunday


def test_trading_days_between_counts_exclusive_start_inclusive_end() -> None:
    # (Jan 1 2021 Fri = New Year holiday, excluded as start) → Jan 2–8: trading days Jan 4,5,6,7,8 = 5.
    assert trading_days_between(date(2021, 1, 1), date(2021, 1, 8)) == 5
    # Empty / reversed ranges are 0.
    assert trading_days_between(date(2021, 6, 15), date(2021, 6, 15)) == 0
    assert trading_days_between(date(2021, 6, 20), date(2021, 6, 15)) == 0


def test_trading_days_between_reflects_the_juneteenth_gate() -> None:
    # (Jun 17 2021, Jun 21 2021]: Fri 18 (now a trading day, pre-2022), Sat 19, Sun 20, Mon 21 → 2.
    # With the old all-years gate, Fri 18 would have been a holiday and this would be 1.
    assert trading_days_between(date(2021, 6, 17), date(2021, 6, 21)) == 2
    # In 2022 the window straddling observed-Juneteenth loses Monday 20: (Jun 16, Jun 21] →
    # Fri 17, Mon 20 (holiday), Tue 21 → Fri 17 + Tue 21 = 2.
    assert trading_days_between(date(2022, 6, 16), date(2022, 6, 21)) == 2
