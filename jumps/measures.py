"""
Pure-Python reference for the jump_daily measures (no ClickHouse, no NumPy) — the oracle the
set-based SQL in jump_daily.py mirrors, and what tests/jumps/test_measures.py checks. Definitions
match jump_daily.py exactly; every undefined case is None (the NULL the SQL writes), never 0 — except
jump_var/cont_var, where 0 is the real "no jump" value.

Intraday (RV = rv_intraday, BV = bv_intraday, TQ = tq_intraday, n = n_bars):
  jump_z = ((RV − BV)/RV) / sqrt(θ · (1/n) · max(1, TQ/BV²)),  θ = (π/2)² + π − 5
           None when BV or TQ is None, or RV = 0, or BV = 0.
  intraday_jump = 1 if jump_z > Z_INTRADAY else 0;  None when jump_z is None.
  jump_var = max(RV − BV, 0) if intraday_jump == 1 else 0;  cont_var = RV − jump_var;  both None when
             intraday_jump is None.
Overnight (signed overnight log return o; rv_overnight = o²):
  overnight_z = |o| / sqrt(median of the trailing sessions' o², excluding today);  None when fewer
                than ON_MIN_HISTORY of them, or the median is 0. The trailing sessions are the name's
                previous rv_daily sessions, at most ON_WINDOW of them, and only those within
                ON_MAX_AGE_DAYS calendar days of the row (so a name returning after a long gap gets
                None until it has ON_MIN_HISTORY recent sessions).
  overnight_jump = 1 if overnight_z >= Z_OVERNIGHT else 0;  None when overnight_z is None.
  overnight_ret_adj = o − SPY's o that date;  None for SPY itself and when SPY has no row.
  overnight_z_adj / overnight_jump_adj: the same rule on overnight_ret_adj, median of its own o_adj².
"""

from __future__ import annotations

import datetime as dt
import math
from statistics import median as _median
from typing import Optional, Sequence

from . import ON_MAX_AGE_DAYS, ON_MIN_HISTORY, ON_WINDOW, THETA, Z_INTRADAY, Z_OVERNIGHT


def intraday_jump_z(rv: float, bv: Optional[float], tq: Optional[float], n: int) -> Optional[float]:
    """The BNS ratio statistic, or None when it is undefined (missing BV/TQ, RV = 0, or BV = 0)."""
    if bv is None or tq is None or rv == 0 or bv == 0:
        return None
    ratio = (rv - bv) / rv
    scale = math.sqrt(THETA * (1.0 / n) * max(1.0, tq / (bv * bv)))
    return ratio / scale


def intraday_flag(z: Optional[float]) -> Optional[int]:
    if z is None:
        return None
    return 1 if z > Z_INTRADAY else 0


def jump_cont_var(rv: float, bv: Optional[float], flag: Optional[int]
                  ) -> tuple[Optional[float], Optional[float]]:
    """(jump_var, cont_var). Both None when the flag is None; otherwise jump_var is max(RV−BV,0) on a
    flagged day and 0 on an unflagged one, and cont_var is RV − jump_var (so they sum to RV)."""
    if flag is None:
        return (None, None)
    jump_var = max(rv - bv, 0.0) if flag == 1 else 0.0  # bv is non-None whenever flag is non-None
    return (jump_var, rv - jump_var)


def recent_sq(dated_sqs: Sequence[tuple], today: dt.date, window: int = ON_WINDOW,
              max_age: int = ON_MAX_AGE_DAYS) -> list[float]:
    """The squared returns that feed the overnight median: from `dated_sqs` (ascending (date, o²)
    pairs), the sessions strictly before `today` (excludes today), the most recent `window` of them,
    then only those within `max_age` calendar days of `today`. Taking the most-recent-window and then
    age-filtering is exact — it equals age-filtering first and then the most-recent-window, because
    any session dropped by age is older than every session kept."""
    prev = [(d, s) for (d, s) in dated_sqs if d < today]
    prev = prev[-window:]
    return [s for (d, s) in prev if (today - d).days <= max_age]


def overnight_z(ret: Optional[float], prev_sqs: Sequence[float]) -> Optional[float]:
    """|ret| over sqrt(median of the previous squared returns). None when `ret` is None, when there
    are fewer than ON_MIN_HISTORY previous sessions, or when the median is 0."""
    if ret is None:
        return None
    if len(prev_sqs) < ON_MIN_HISTORY:
        return None
    m = _median(prev_sqs)
    if m == 0:
        return None
    return abs(ret) / math.sqrt(m)


def overnight_flag(z: Optional[float]) -> Optional[int]:
    if z is None:
        return None
    return 1 if z >= Z_OVERNIGHT else 0


def overnight_ret_adj(name_ret: Optional[float], spy_ret: Optional[float],
                      is_spy: bool) -> Optional[float]:
    """The market-adjusted overnight move: the name's overnight return less SPY's that date. None for
    SPY itself (nothing to adjust against) and None when SPY has no row that date."""
    if is_spy or name_ret is None or spy_ret is None:
        return None
    return name_ret - spy_ret
