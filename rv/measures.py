"""
Pure-Python reference for the rv_daily intraday measures (realized variance, downside/upside
realized semivariance, bipower variation, tri-power quarticity). No ClickHouse, no NumPy — this is
the offline oracle the set-based SQL in rv_daily.py is checked against, by tests/rv/test_measures.py
and by `python -m rv check`.

Definitions match rv_daily.py exactly (one symbol, one complete day). Let r_1..r_n be the intraday
5-minute log returns of grid_px (the 09:30 open followed by the bucket closes) — the successive
differences of log(grid_px) with the leading grid point dropped; n = n_bars. Let o be the signed
overnight log return (rv_overnight = o²).

  rv_intraday = Σ r_i²
  rs_down     = Σ_{r_i<0} r_i²  + (o² if o < 0)
  rs_up       = Σ_{r_i>0} r_i²  + (o² if o > 0)   → rs_down + rs_up = rv_intraday + o² = rv_total
  bv_intraday = (π/2)·(n/(n−1))·Σ_{i=2..n} |r_i|·|r_{i−1}|                         (None when n < 2)
  tq_intraday = n·μ⁻³·(n/(n−2))·Σ_{i=3..n} |r_i|^(4/3)·|r_{i−1}|^(4/3)·|r_{i−2}|^(4/3)  (None when n < 3)

with μ = 2^(2/3)·Γ(7/6)/Γ(1/2) ≈ 0.8309. Zero returns go in neither semivariance bucket. No value is
ever defaulted to 0: when a measure is undefined (n below its minimum) it is None, matching the
NULL the SQL writes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

# μ = 2^(2/3)·Γ(7/6)/Γ(1/2) ≈ 0.8309 — the bipower/tri-power scaling constant (same formula as
# rv_daily._MU; computed from math.gamma so the value is never transcribed by hand).
MU = 2 ** (2 / 3) * math.gamma(7 / 6) / math.gamma(1 / 2)


class DayMeasures(NamedTuple):
    rv_intraday: float
    rs_down: float
    rs_up: float
    bv_intraday: float | None
    tq_intraday: float | None


def day_measures(grid_px: Sequence[float], overnight_ret: float) -> DayMeasures:
    """The five intraday measures for one day. `grid_px` is the ordered grid prices with the 09:30
    open first (n+1 points → n intraday returns); `overnight_ret` is the signed overnight log return
    (rv_overnight = overnight_ret²)."""
    logpx = [math.log(p) for p in grid_px]
    rets = [logpx[i + 1] - logpx[i] for i in range(len(logpx) - 1)]  # r_1..r_n (leading point dropped)
    n = len(rets)

    rv_intraday = sum(r * r for r in rets)
    o2 = overnight_ret * overnight_ret
    rs_down = sum(r * r for r in rets if r < 0) + (o2 if overnight_ret < 0 else 0.0)
    rs_up = sum(r * r for r in rets if r > 0) + (o2 if overnight_ret > 0 else 0.0)

    bv_intraday: float | None = None
    if n >= 2:
        s = sum(abs(rets[i]) * abs(rets[i - 1]) for i in range(1, n))
        bv_intraday = (math.pi / 2) * (n / (n - 1)) * s

    tq_intraday: float | None = None
    if n >= 3:
        s = sum(
            abs(rets[i]) ** (4 / 3) * abs(rets[i - 1]) ** (4 / 3) * abs(rets[i - 2]) ** (4 / 3)
            for i in range(2, n)
        )
        tq_intraday = n * MU ** (-3) * (n / (n - 2)) * s

    return DayMeasures(rv_intraday, rs_down, rs_up, bv_intraday, tq_intraday)
