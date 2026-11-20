"""
The pure window math of method fwd_v1 — no ClickHouse, no clock, no calendar of its own. Everything
that touches the calendar, the day weights or the earnings tables is handed in as a callable, so the
arithmetic here is deterministic and unit-tested in isolation (mirrors iv_curve/iso.py).

Per (underlying, session_date), given the ok iso_v1 expiry points:

  - select_nodes: keep expiries with enough support, by τ; a run of consecutive equal fitted total
    variance is one pooled block, of which only the LAST expiry is a node (a zero-variance window
    inside a pooled block carries no information — Stage 2 check lesson);
  - build_day: the windows between consecutive nodes (the anchor at τ=0, variance 0, is the first
    lower bound). Each window's trading days (after the start, up to and including the end) are summed
    into effective days by their gap weight; window variance is the fitted-variance step; the day's
    normal per-effective-day variance is the median over its windows; excess is what each window holds
    beyond normal, and √(excess) is the implied ±move. Fewer than MIN_WINDOWS windows → one row with
    status insufficient_windows and no excess.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Callable, List, Optional, Sequence, Tuple

from . import (
    EARNINGS_HISTORY, MIN_NODE_SUPPORT, MIN_WINDOWS, STATUS_INSUFFICIENT_WINDOWS, STATUS_OK,
)


@dataclass(frozen=True)
class ExpiryNode:
    """One ok iso_v1 expiry point: τ trading days to it, the fitted total variance, and the support
    (Σ n_trades of its pooled block) — the curve's own numbers, read verbatim."""

    expiry: date
    tau: int
    fit_total_var: float
    support: float


@dataclass(frozen=True)
class Window:
    """One output row. For an ok window every field is set; for the single insufficient_windows row
    the expiries/variance/support/earnings are None and start_td = end_td = 0, eff_days = 0."""

    start_expiry: Optional[date]
    end_expiry: Optional[date]
    start_td: int
    end_td: int
    eff_days: float
    window_var: Optional[float]
    normal_daily_var: Optional[float]
    excess_var: Optional[float]
    excess_days: Optional[float]
    implied_move: Optional[float]
    support: Optional[float]
    own_earnings_date: Optional[date]
    own_earnings_source: Optional[str]
    status: str


@dataclass(frozen=True)
class _Raw:
    """A window before the day's normal is known (window variance and effective days measured)."""

    start_expiry: Optional[date]
    end_expiry: date
    start_td: int
    end_td: int
    eff_days: float
    window_var: float
    support: float
    own_earnings_date: Optional[date]
    own_earnings_source: Optional[str]


def select_nodes(expiries: Sequence[ExpiryNode], min_support: float = MIN_NODE_SUPPORT
                 ) -> List[ExpiryNode]:
    """Expiries with support >= min_support, ordered by τ; a run of consecutive equal fit_total_var is
    one pooled block, collapsed to its LAST expiry (the only node that adds a non-zero window)."""
    kept = sorted((e for e in expiries if e.support >= min_support), key=lambda e: e.tau)
    nodes: List[ExpiryNode] = []
    for e in kept:
        if nodes and nodes[-1].fit_total_var == e.fit_total_var:
            nodes[-1] = e   # same pooled block → advance the node to the block's last expiry
        else:
            nodes.append(e)
    return nodes


def _raw_windows(session_date: date, nodes: Sequence[ExpiryNode], *,
                 day_weight: Callable[[date], float],
                 list_days: Callable[[date, date], List[date]],
                 own_earnings: Callable[[date, date], Optional[Tuple[date, str]]]) -> List[_Raw]:
    raws: List[_Raw] = []
    prev_expiry: Optional[date] = None   # the anchor at τ=0
    prev_tau = 0
    prev_var = 0.0
    prev_support = math.inf
    for n in nodes:
        start = prev_expiry if prev_expiry is not None else session_date
        days = list_days(start, n.expiry)           # after the start, up to and incl. the end
        eff_days = math.fsum(day_weight(d) for d in days)
        oe = own_earnings(start, n.expiry)          # (start, end]: start exclusive, end inclusive
        raws.append(_Raw(
            start_expiry=prev_expiry, end_expiry=n.expiry, start_td=prev_tau, end_td=n.tau,
            eff_days=eff_days, window_var=n.fit_total_var - prev_var,
            support=min(prev_support, n.support),
            own_earnings_date=oe[0] if oe is not None else None,
            own_earnings_source=oe[1] if oe is not None else None))
        prev_expiry, prev_tau, prev_var, prev_support = n.expiry, n.tau, n.fit_total_var, n.support
    return raws


def _insufficient() -> Window:
    return Window(
        start_expiry=None, end_expiry=None, start_td=0, end_td=0, eff_days=0.0, window_var=None,
        normal_daily_var=None, excess_var=None, excess_days=None, implied_move=None, support=None,
        own_earnings_date=None, own_earnings_source=None, status=STATUS_INSUFFICIENT_WINDOWS)


def build_day(session_date: date, nodes: Sequence[ExpiryNode], *,
              day_weight: Callable[[date], float],
              list_days: Callable[[date, date], List[date]],
              own_earnings: Callable[[date, date], Optional[Tuple[date, str]]]) -> List[Window]:
    """All window rows for one (underlying, session_date). Fewer than MIN_WINDOWS windows → a single
    insufficient_windows row. Otherwise the day's normal is the median per-effective-day variance, and
    each window's excess is measured against it (excess may be negative; the implied move clamps at 0)."""
    raws = _raw_windows(session_date, nodes, day_weight=day_weight, list_days=list_days,
                        own_earnings=own_earnings)
    if len(raws) < MIN_WINDOWS:
        return [_insufficient()]
    normal = statistics.median(r.window_var / r.eff_days for r in raws)
    out: List[Window] = []
    for r in raws:
        excess_var = r.window_var - normal * r.eff_days
        out.append(Window(
            start_expiry=r.start_expiry, end_expiry=r.end_expiry, start_td=r.start_td,
            end_td=r.end_td, eff_days=r.eff_days, window_var=r.window_var, normal_daily_var=normal,
            excess_var=excess_var, excess_days=excess_var / normal,
            implied_move=math.sqrt(max(excess_var, 0.0)), support=r.support,
            own_earnings_date=r.own_earnings_date, own_earnings_source=r.own_earnings_source,
            status=STATUS_OK))
    return out


# ── earnings attribution (pure) ──────────────────────────────────────────────────────────────────

def history_reaction_session(reported_date: date, reaction_date: Optional[date]) -> date:
    """The session an earnings_history report reacts on: its measured reaction_date, or the reported
    date when the reaction day could not be found."""
    return reaction_date if reaction_date is not None else reported_date


def calendar_reaction_session(day: date, earnings_et: Optional[datetime], *,
                              session_close_et: Callable[[date], time],
                              is_trading_day: Callable[[date], bool],
                              next_trading_day: Callable[[date], date]) -> date:
    """The session a future (earnings_calendar) report reacts on: the next trading day when the report
    is timestamped after that day's session close, else the day itself — advanced to the next trading
    day if `day` is not itself a session. A missing timestamp is treated as on/before the close."""
    after_close = earnings_et is not None and earnings_et > datetime.combine(day, session_close_et(day))
    if after_close or not is_trading_day(day):
        return next_trading_day(day)
    return day


def earliest_in_window(candidates: Sequence[Tuple[date, str]], start_exclusive: date,
                       end_inclusive: date) -> Optional[Tuple[date, str]]:
    """The earliest (reaction_session, source) strictly after start and up to and including end; on a
    tie earnings_history wins. None if nothing falls in the window."""
    in_win = [c for c in candidates if start_exclusive < c[0] <= end_inclusive]
    if not in_win:
        return None
    return min(in_win, key=lambda c: (c[0], 0 if c[1] == EARNINGS_HISTORY else 1))
