"""
The measured day-weight calibration (method fwd_v1). A gap is the number of calendar days since the
previous trading day; gaps of 5+ pool into one >=5 bucket. For each gap that occurs in the range, the
weight is the median rv_total (pooled across names) divided by the gap-1 median — so a gap-1 day is 1
by definition, a weekend (gap 3) and a post-holiday day (gap 4) carry more, measured not assumed. A
gap that occurs with fewer than MIN_DAYTYPE_N observations is an error naming it; a gap that never
occurs simply gets no row. Writes trading.iv_event_daytype.
"""

from __future__ import annotations

import datetime as dt
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from ml.shared.calendar import previous_trading_day

from . import (
    GAP_BASE, GAP_MAX_BUCKET, IV_EVENT_DAYTYPE_TABLE, METHOD, MIN_DAYTYPE_N, RV_DAILY_TABLE, RV_SOURCE,
)

_DAYTYPE_COLUMNS = ("method", "gap_days", "weight", "n", "from_date", "to_date")


@dataclass(frozen=True)
class DaytypeRow:
    gap_days: int
    weight: float
    n: int


def gap_bucket(d: dt.date) -> int:
    """The gap (calendar days since the previous trading day) for `d`, pooled at GAP_MAX_BUCKET."""
    return min((d - previous_trading_day(d)).days, GAP_MAX_BUCKET)


def compute_daytype(values_by_gap: Mapping[int, Sequence[float]],
                    min_n: int = MIN_DAYTYPE_N) -> List[DaytypeRow]:
    """One row per gap that occurs, weight = its median rv_total / the gap-1 median. Errors naming a
    gap that occurs with fewer than `min_n` observations, or if gap 1 is absent (can't normalise)."""
    if GAP_BASE not in values_by_gap or len(values_by_gap[GAP_BASE]) == 0:
        raise ValueError(f"no rv_total observations at gap {GAP_BASE} in the range - cannot normalise "
                         "the day weights (widen the range)")
    for gap in sorted(values_by_gap):
        n = len(values_by_gap[gap])
        if n < min_n:
            raise ValueError(f"gap {gap} occurs with only {n} rv_total observations (< {min_n}) - "
                             "widen the date range")
    base = statistics.median(values_by_gap[GAP_BASE])
    if base <= 0:
        raise ValueError(f"median rv_total at gap {GAP_BASE} is {base} - cannot normalise")
    return [DaytypeRow(gap_days=gap, weight=statistics.median(values_by_gap[gap]) / base,
                       n=len(values_by_gap[gap]))
            for gap in sorted(values_by_gap)]


def _rv_by_gap(ch: Any, start: dt.date, end: dt.date) -> Dict[int, List[float]]:
    res = ch.query(
        f"SELECT trading_date, rv_total FROM {RV_DAILY_TABLE} FINAL "
        "WHERE source = %(src)s AND rv_total > 0 AND isFinite(rv_total) "
        "AND trading_date BETWEEN %(a)s AND %(b)s",
        parameters={"src": RV_SOURCE, "a": start, "b": end})
    by_gap: Dict[int, List[float]] = defaultdict(list)
    for trading_date, rv in res.result_rows:
        by_gap[gap_bucket(trading_date)].append(float(rv))
    return by_gap


def _write(ch: Any, rows: Sequence[DaytypeRow], start: dt.date, end: dt.date) -> None:
    out = [[METHOD, r.gap_days, r.weight, r.n, start, end] for r in rows]
    ch.insert(IV_EVENT_DAYTYPE_TABLE, out, column_names=list(_DAYTYPE_COLUMNS))


def run(ch: Any, start: dt.date, end: dt.date) -> List[DaytypeRow]:
    """Measure the day weights over [start, end] (pooled across names), write them, and return them."""
    rows = compute_daytype(_rv_by_gap(ch, start, end))
    _write(ch, rows, start, end)
    return rows


def print_table(rows: Sequence[DaytypeRow], start: dt.date, end: dt.date) -> None:
    print(f"iv_event daytype {METHOD}  {start} .. {end}")
    print(f"  {'gap':>4s} {'weight':>10s} {'n':>12s}")
    for r in rows:
        label = f">={r.gap_days}" if r.gap_days == GAP_MAX_BUCKET else str(r.gap_days)
        print(f"  {label:>4s} {r.weight:>10.4f} {r.n:>12d}")
