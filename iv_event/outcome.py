"""
The pure outcome math of method out_v1 — no ClickHouse, no clock, no calendar of its own. Everything
that touches the calendar, the realized/jump/curve data or the corporate-action/market-wide sets is
handed in as data or a callable, so the arithmetic here is deterministic and unit-tested in isolation
(mirrors fwd.py).

Per passed iv_event_window row (method fwd_v1, status ok), given its underlying, session_date s,
start_expiry S (None = anchor), end_expiry E, end_td, eff_days and normal_daily_var:

  - window_days: the trading days d with max(s, S) < d <= E (anchor: s < d <= E) — the same days the
    window's eff_days was summed over;
  - realized: n_rv_days = window days with an rv_daily row; realized_var = Σ rv_total over them, but
    NULL unless every window day has a row (no partial sums); realized_excess_var = realized_var −
    normal_daily_var·eff_days (NULL when realized_var is);
  - jump_stats: n_jump_days / first_jump_date over window days flagged overnight_jump_adj=1 or
    intraday_jump=1; max_abs_ret_adj = the largest |overnight_ret_adj| (NULL if all NULL);
  - releases: for each window day d, release(d) = TV(d_prev, E) − TV(d, E) with TV(E, E) = 0, present
    only when both values exist; excess_release(d) = release(d) − normal_daily_var·w(d) with w the
    day's daytype weight; crush = the largest excess_release, earliest on a tie;
  - corporate_action_in_window: a non-dividend/name-change corp action or a split in [s, E];
  - explained: 'own_earnings' → 'market_wide' → 'none', in that precedence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Optional, Sequence

# The three explained values, in precedence order.
EXPLAINED_OWN_EARNINGS = "own_earnings"
EXPLAINED_MARKET_WIDE = "market_wide"
EXPLAINED_NONE = "none"


@dataclass(frozen=True)
class WindowSpec:
    """One passed window's inputs, read verbatim from an iv_event_window row."""

    underlying: str
    session_date: date
    start_expiry: Optional[date]   # S; None for the anchor window
    end_expiry: date               # E
    end_td: int
    eff_days: float
    normal_daily_var: float
    own_earnings_date: Optional[date]


@dataclass(frozen=True)
class Outcome:
    """The computed outcome fields for one window (everything but its provenance/identity columns)."""

    n_days: int
    n_rv_days: int
    realized_var: Optional[float]
    realized_excess_var: Optional[float]
    n_jump_days: int
    first_jump_date: Optional[date]
    max_abs_ret_adj: Optional[float]
    n_release_days: int
    crush_date: Optional[date]
    crush_excess_var: Optional[float]
    corporate_action: int
    explained: str


@dataclass(frozen=True)
class JumpRow:
    """A window day's jump_daily row, the only three fields the outcome uses (None = NULL)."""

    overnight_jump_adj: Optional[int]
    intraday_jump: Optional[int]
    overnight_ret_adj: Optional[float]


def window_days(s: date, start_expiry: Optional[date], end_expiry: date,
                list_days: Callable[[date, date], List[date]]) -> List[date]:
    """The trading days d with max(s, S) < d <= E (anchor, S None: s < d <= E). `list_days` is the
    same (start_exclusive, end_inclusive] trading-day lister the window used for eff_days."""
    start_exclusive = s if start_expiry is None else max(s, start_expiry)
    return list_days(start_exclusive, end_expiry)


def realized(days: Sequence[date],
             rv_by_date: Dict[date, float]) -> "tuple[int, Optional[float]]":
    """(n_rv_days, realized_var). realized_var = Σ rv_total over the window days, but NULL unless every
    window day has a row — no partial sums (and NULL for an empty window, never a fabricated 0 sum)."""
    present = [rv_by_date[d] for d in days if d in rv_by_date]
    n_rv_days = len(present)
    realized_var = sum(present) if (days and n_rv_days == len(days)) else None
    return n_rv_days, realized_var


def jump_stats(days: Sequence[date],
               jump_by_date: Dict[date, JumpRow]
               ) -> "tuple[int, Optional[date], Optional[float]]":
    """(n_jump_days, first_jump_date, max_abs_ret_adj). A window day counts as a jump day when its
    jump_daily row has overnight_jump_adj = 1 OR intraday_jump = 1; max_abs_ret_adj is the largest
    |overnight_ret_adj| over the window days with a non-NULL value (None if all NULL/absent)."""
    jump_dates: List[date] = []
    abs_adj: List[float] = []
    for d in days:
        row = jump_by_date.get(d)
        if row is None:
            continue
        if row.overnight_jump_adj == 1 or row.intraday_jump == 1:
            jump_dates.append(d)
        if row.overnight_ret_adj is not None:
            abs_adj.append(abs(row.overnight_ret_adj))
    first_jump_date = min(jump_dates) if jump_dates else None
    max_abs = max(abs_adj) if abs_adj else None
    return len(jump_dates), first_jump_date, max_abs


def releases(days: Sequence[date], end_expiry: date, normal_daily_var: float,
             prev_trading_day: Callable[[date], date],
             curve_tv: Callable[[date, date], Optional[float]],
             day_weight: Callable[[date], float]
             ) -> "tuple[int, Optional[date], Optional[float]]":
    """(n_release_days, crush_date, crush_excess_var). For each window day d, release(d) =
    TV(d_prev, E) − TV(d, E) with TV(E, E) = 0 and TV read from iv_curve_daily (kind expiry, this
    expiry) otherwise; a day has a release only when BOTH TV values exist. excess_release(d) =
    release(d) − normal_daily_var·w(d). crush = the largest excess_release; earliest on a tie."""
    def tv(session: date) -> Optional[float]:
        return 0.0 if session == end_expiry else curve_tv(session, end_expiry)

    n_release = 0
    crush_date: Optional[date] = None
    crush_excess: Optional[float] = None
    for d in days:   # ascending → strict `>` below keeps the earliest on a tie
        tv_prev = tv(prev_trading_day(d))
        tv_d = tv(d)
        if tv_prev is None or tv_d is None:
            continue
        n_release += 1
        excess = (tv_prev - tv_d) - normal_daily_var * day_weight(d)
        if crush_excess is None or excess > crush_excess:
            crush_excess, crush_date = excess, d
    return n_release, crush_date, crush_excess


def corporate_action_in_window(ca_dates: Sequence[date], split_dates: Sequence[date],
                               s: date, end_expiry: date) -> int:
    """1 when any corporate-action date (coalesce(ex_date, event_date) of a kind already filtered to
    exclude cash_dividend/name_change) or any split execution_date falls in [s, E]; else 0."""
    for d in ca_dates:
        if s <= d <= end_expiry:
            return 1
    for d in split_dates:
        if s <= d <= end_expiry:
            return 1
    return 0


def explained(own_earnings_date: Optional[date], first_jump_date: Optional[date],
              crush_date: Optional[date], is_market_wide: Callable[[date], bool]) -> str:
    """'own_earnings' when the window has an own earnings reaction in it; else 'market_wide' when its
    first_jump_date or crush_date is a market-wide date; else 'none'. Precedence is fixed."""
    if own_earnings_date is not None:
        return EXPLAINED_OWN_EARNINGS
    for d in (first_jump_date, crush_date):
        if d is not None and is_market_wide(d):
            return EXPLAINED_MARKET_WIDE
    return EXPLAINED_NONE


def build_outcome(spec: WindowSpec, *,
                  list_days: Callable[[date, date], List[date]],
                  prev_trading_day: Callable[[date], date],
                  day_weight: Callable[[date], float],
                  rv_by_date: Dict[date, float],
                  jump_by_date: Dict[date, JumpRow],
                  curve_tv: Callable[[date, date], Optional[float]],
                  ca_dates: Sequence[date],
                  split_dates: Sequence[date],
                  is_market_wide: Callable[[date], bool]) -> "tuple[List[date], Outcome]":
    """Assemble one window's Outcome from the pure pieces. Returns (window_days, Outcome); the caller
    uses the day list for the span-coverage check and the insert. realized_excess_var = realized_var −
    normal_daily_var·eff_days, NULL when realized_var is."""
    days = window_days(spec.session_date, spec.start_expiry, spec.end_expiry, list_days)
    n_rv_days, realized_var = realized(days, rv_by_date)
    realized_excess_var = (None if realized_var is None
                           else realized_var - spec.normal_daily_var * spec.eff_days)
    n_jump_days, first_jump_date, max_abs_ret_adj = jump_stats(days, jump_by_date)
    n_release_days, crush_date, crush_excess_var = releases(
        days, spec.end_expiry, spec.normal_daily_var, prev_trading_day, curve_tv, day_weight)
    corp = corporate_action_in_window(ca_dates, split_dates, spec.session_date, spec.end_expiry)
    expl = explained(spec.own_earnings_date, first_jump_date, crush_date, is_market_wide)
    return days, Outcome(
        n_days=len(days), n_rv_days=n_rv_days, realized_var=realized_var,
        realized_excess_var=realized_excess_var, n_jump_days=n_jump_days,
        first_jump_date=first_jump_date, max_abs_ret_adj=max_abs_ret_adj,
        n_release_days=n_release_days, crush_date=crush_date, crush_excess_var=crush_excess_var,
        corporate_action=corp, explained=expl)
