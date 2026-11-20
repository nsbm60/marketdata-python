"""The measured day weights: the median/normalise computation, the minimum-n error, and gap bucketing."""

from __future__ import annotations

from datetime import date

import pytest

from iv_event import GAP_MAX_BUCKET, MIN_DAYTYPE_N
from iv_event.daytype import compute_daytype, gap_bucket


def test_weights_are_the_per_gap_median_normalised_to_gap_one() -> None:
    values = {1: [1.0] * MIN_DAYTYPE_N, 3: [1.5] * MIN_DAYTYPE_N, 4: [2.0] * (MIN_DAYTYPE_N + 50)}
    rows = {r.gap_days: r for r in compute_daytype(values)}

    assert set(rows) == {1, 3, 4}                 # only gaps that occur get a row — no gap 2 or 5
    assert rows[1].weight == 1.0 and rows[1].n == MIN_DAYTYPE_N
    assert rows[3].weight == 1.5
    assert rows[4].weight == 2.0 and rows[4].n == MIN_DAYTYPE_N + 50


def test_a_gap_below_min_n_is_an_error_naming_it() -> None:
    values = {1: [1.0] * MIN_DAYTYPE_N, 2: [1.0] * (MIN_DAYTYPE_N - 1)}
    with pytest.raises(ValueError, match=r"gap 2 .*< 200"):
        compute_daytype(values)


def test_missing_gap_one_cannot_normalise() -> None:
    with pytest.raises(ValueError, match=r"gap 1"):
        compute_daytype({3: [1.5] * MIN_DAYTYPE_N})


def test_gap_bucket_from_the_real_calendar() -> None:
    assert gap_bucket(date(2021, 1, 12)) == 1     # an ordinary Tuesday (prev day Monday)
    assert gap_bucket(date(2021, 1, 11)) == 3     # a Monday (prev trading day the Friday before)
    assert gap_bucket(date(2021, 1, 19)) == 4     # Tue after the Mon-18 MLK holiday (prev day Fri 15)
    assert gap_bucket(date(2026, 1, 20)) <= GAP_MAX_BUCKET  # always pooled at the >=5 bucket
