"""
Read-only inspection: `python -m iv_event show <date> <underlying>` and
`python -m iv_event top <from> <to> [--min-support N]`. Neither writes anything.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Optional

from . import (
    IV_EVENT_OUTCOME_TABLE, IV_EVENT_TABLE, METHOD, OUTCOME_METHOD, SHOW_FLAG_EXCESS_DAYS,
    SHOW_FLAG_SUPPORT, STATUS_OK, TRADING_DAYS_PER_YEAR, WINDOW_METHOD,
)


def _var_vol_pct(var: Optional[float]) -> str:
    """A variance rendered as an annualised-vol percentage (√var · 100)."""
    return f"{math.sqrt(var) * 100:>7.2f}%" if var is not None and var >= 0 else f"{'—':>8}"


def _move_pct(move: Optional[float]) -> str:
    """An implied ±move (already a return) rendered as a percentage."""
    return f"{move * 100:>7.2f}%" if move is not None else f"{'—':>8}"


def show(ch: Any, day: dt.date, underlying: str) -> None:
    """One line per window for (day, underlying): expiries, trading days, effective days, the per-day
    forward vol (annualised on 252), excess_days, the implied ±move (as %), support and own earnings.
    A window with excess_days >= SHOW_FLAG_EXCESS_DAYS and support >= SHOW_FLAG_SUPPORT is marked EVENT.
    Read-only."""
    res = ch.query(
        f"SELECT start_expiry, end_expiry, start_td, end_td, eff_days, window_var, excess_days, "
        f"implied_move, support, own_earnings_date, own_earnings_source, status "
        f"FROM {IV_EVENT_TABLE} FINAL "
        "WHERE method = %(m)s AND underlying = %(u)s AND session_date = %(d)s "
        "ORDER BY end_td",
        parameters={"m": METHOD, "u": underlying.upper(), "d": day})

    print(f"iv_event {METHOD}  {underlying.upper()}  session {day}")
    rows = res.result_rows
    if not rows:
        print("  (no rows)")
        return

    print(f"    {'start':10s} {'end':10s} {'td':>4s} {'eff':>7s} {'day_vol':>8s} "
          f"{'exc_days':>9s} {'move±':>8s} {'support':>9s}  earnings")
    for (start_e, end_e, start_td, end_td, eff, wvar, exc_days, move, support,
         oe_date, oe_src, status) in rows:
        if status != STATUS_OK:
            print(f"    {status}")
            continue
        td = int(end_td) - int(start_td)
        per_day_vol = ((float(wvar) / float(eff)) * TRADING_DAYS_PER_YEAR
                       if wvar is not None and eff and float(eff) > 0 else None)
        earn = f"{oe_date} ({oe_src})" if oe_date is not None else "—"
        flag = (" EVENT" if exc_days is not None and support is not None
                and float(exc_days) >= SHOW_FLAG_EXCESS_DAYS and float(support) >= SHOW_FLAG_SUPPORT
                else "")
        start_s = str(start_e) if start_e is not None else "anchor"
        print(f"    {start_s:10s} {str(end_e):10s} {td:>4d} {float(eff):>7.2f} "
              f"{_var_vol_pct(per_day_vol)} "
              f"{(float(exc_days) if exc_days is not None else float('nan')):>9.2f} "
              f"{_move_pct(float(move) if move is not None else None)} "
              f"{(float(support) if support is not None else float('nan')):>9.1f}  {earn}{flag}")


def outcome_show(ch: Any, day: dt.date, underlying: str) -> None:
    """One line per window for (day, underlying): the window beside its out_v1 outcome (LEFT JOIN, so
    a not-yet-passed window shows a blank outcome). Read-only."""
    res = ch.query(
        f"SELECT w.start_expiry, w.end_expiry, w.end_td, w.excess_days, w.support, w.status, "
        f"o.n_days, o.n_rv_days, o.realized_var, o.realized_excess_var, o.n_jump_days, "
        f"o.first_jump_date, o.max_abs_ret_adj, o.n_release_days, o.crush_date, o.crush_excess_var, "
        f"o.corporate_action, o.explained "
        f"FROM (SELECT * FROM {IV_EVENT_TABLE} FINAL "
        f"      WHERE method = %(m)s AND underlying = %(u)s AND session_date = %(d)s) w "
        f"LEFT JOIN (SELECT * FROM {IV_EVENT_OUTCOME_TABLE} FINAL WHERE method = %(om)s "
        f"           AND underlying = %(u)s AND session_date = %(d)s) o "
        f"  ON w.end_td = o.end_td "
        f"ORDER BY w.end_td "
        f"SETTINGS join_use_nulls = 1",
        parameters={"m": WINDOW_METHOD, "om": OUTCOME_METHOD, "u": underlying.upper(), "d": day})

    print(f"iv_event outcome {OUTCOME_METHOD}  {underlying.upper()}  session {day}")
    rows = res.result_rows
    if not rows:
        print("  (no rows)")
        return
    print(f"    {'start':10s} {'end':10s} {'td':>4s} {'exc_d':>7s} {'n_d':>4s} {'n_rv':>5s} "
          f"{'realVar':>10s} {'realXS':>10s} {'nJ':>3s} {'firstJump':>11s} {'crush':>11s} "
          f"{'crushXS':>10s} {'corp':>4s} {'expl':>12s}")
    for (se, ee, end_td, exc_d, support, status, nd, nrv, rv, rxv, njd, fjd, mar, nrd, cd, cxv,
         corp, expl) in rows:
        if status != STATUS_OK:
            print(f"    {status}")
            continue
        start_s = str(se) if se is not None else "anchor"
        if nd is None:   # window exists but not yet passed → no outcome row
            print(f"    {start_s:10s} {str(ee):10s} {int(end_td):>4d} "
                  f"{(float(exc_d) if exc_d is not None else float('nan')):>7.2f}  (not passed)")
            continue
        print(f"    {start_s:10s} {str(ee):10s} {int(end_td):>4d} "
              f"{(float(exc_d) if exc_d is not None else float('nan')):>7.2f} "
              f"{int(nd):>4d} {int(nrv):>5d} {_var(rv):>10s} {_var(rxv):>10s} {int(njd):>3d} "
              f"{str(fjd) if fjd is not None else '—':>11s} {str(cd) if cd is not None else '—':>11s} "
              f"{_var(cxv):>10s} {int(corp):>4d} {str(expl):>12s}")


def _var(x: Optional[float]) -> str:
    return "—" if x is None else f"{float(x):.3e}"


def top(ch: Any, start: dt.date, end: dt.date, min_support: float) -> None:
    """The 50 windows with the largest implied_move in [start, end] that have NO own earnings in the
    window and support >= min_support — the candidate non-earnings events. Read-only, writes nothing."""
    res = ch.query(
        f"SELECT session_date, underlying, start_expiry, end_expiry, implied_move, excess_days, support "
        f"FROM {IV_EVENT_TABLE} FINAL "
        "WHERE method = %(m)s AND status = %(ok)s AND session_date BETWEEN %(a)s AND %(b)s "
        "AND own_earnings_date IS NULL AND implied_move IS NOT NULL AND support >= %(ms)s "
        "ORDER BY implied_move DESC LIMIT 50",
        parameters={"m": METHOD, "ok": STATUS_OK, "a": start, "b": end, "ms": min_support})

    print(f"iv_event {METHOD} top non-earnings windows  {start} .. {end}  (support >= {min_support:g})")
    rows = res.result_rows
    if not rows:
        print("  (no rows)")
        return
    print(f"    {'session':10s} {'underlying':12s} {'start':10s} {'end':10s} "
          f"{'move±':>8s} {'exc_days':>9s} {'support':>9s}")
    for session_date, underlying, start_e, end_e, move, exc_days, support in rows:
        start_s = str(start_e) if start_e is not None else "anchor"
        move_pct = f"{float(move) * 100:>7.2f}%"
        print(f"    {str(session_date):10s} {str(underlying):12s} {start_s:10s} {str(end_e):10s} "
              f"{move_pct} {float(exc_days):>9.2f} {float(support):>9.1f}")
