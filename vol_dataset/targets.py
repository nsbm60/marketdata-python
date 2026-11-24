"""
The forward targets and the drop rules — NOT part of the look-ahead-safe input path (targets read the
H trading days AFTER t). Per (underlying, t, H) that exists (an ok curve horizon row and an rv_daily
row for each of the H trading days after t):

  realized_down_var = Σ rs_down over the H days, realized_up_var = Σ rs_up,
  implied_total_var = the horizon row's fit_total_var, sample_weight = its support,
  target_down/target_up = log(2 · realized / implied)  (rules.compute_targets).

A corporate action or split in (t, t+H] drops the row (iv_event's corporate_action rule, reused). A
zero realized sum drops the row. (underlying, t, H) with an ok curve row but incomplete H rv days do
not exist — they are counted as coverage, not drops.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from ml.shared.calendar import trading_days_in

from iv_event.outcome import corporate_action_in_window
from iv_event.outcome_event import read_corp_actions, read_splits

from . import (
    CURVE_KIND_HORIZON, CURVE_METHOD, CURVE_STATUS_OK, FORWARD_CALENDAR_DAYS, HORIZONS_TD,
    IV_CURVE_TABLE, RV_DAILY_TABLE, RV_SOURCE, SPY_SYMBOL,
)
from . import rules

_TARGET_COLUMNS = (
    "underlying", "t", "H", "implied_total_var", "realized_down_var", "realized_up_var",
    "target_down", "target_up", "sample_weight",
)


def _q(ch: Any, sql: str, params: Dict[str, Any]) -> List[tuple]:
    return ch.query(sql, parameters=params).result_rows


def compute(ch: Any, start: dt.date, end: dt.date
            ) -> Tuple[pd.DataFrame, Counter, Counter]:
    """(targets_df, dropped, coverage). targets_df holds the existing, non-dropped rows. `dropped`
    counts drops by reason ('corporate_action' / 'zero_realized_down' / 'zero_realized_up'); `coverage`
    counts ok-curve rows with incomplete H rv days, keyed (year, H)."""
    curve = _q(
        ch,
        f"SELECT underlying, session_date, horizon_td, fit_total_var, support "
        f"FROM {IV_CURVE_TABLE} FINAL "
        f"WHERE method = %(m)s AND kind = %(k)s AND status = %(s)s "
        f"AND session_date BETWEEN %(a)s AND %(b)s AND fit_total_var IS NOT NULL",
        {"m": CURVE_METHOD, "k": CURVE_KIND_HORIZON, "s": CURVE_STATUS_OK, "a": start, "b": end})
    names = sorted({str(r[0]).upper() for r in curve} | {SPY_SYMBOL})

    fwd_hi = end + dt.timedelta(days=FORWARD_CALENDAR_DAYS)
    rv_rows = _q(
        ch,
        f"SELECT symbol, trading_date, rs_down, rs_up FROM {RV_DAILY_TABLE} FINAL "
        f"WHERE source = %(s)s AND symbol IN %(u)s AND trading_date BETWEEN %(a)s AND %(b)s",
        {"s": RV_SOURCE, "u": names, "a": start, "b": fwd_hi})
    rs_down: Dict[Tuple[str, dt.date], float] = {}
    rs_up: Dict[Tuple[str, dt.date], float] = {}
    present: Set[Tuple[str, dt.date]] = set()
    for sym, d, rd, ru in rv_rows:
        key = (str(sym).upper(), _as_date(d))
        present.add(key)
        rs_down[key] = float(rd) if rd is not None else 0.0
        rs_up[key] = float(ru) if ru is not None else 0.0

    uni = {str(r[0]).upper() for r in curve}
    corp = read_corp_actions(ch, uni)
    splits = read_splits(ch, uni)

    dropped: Counter = Counter()
    coverage: Counter = Counter()
    out: List[tuple] = []
    for u, sd, h, ftv, sup in curve:
        u = str(u).upper()
        t = _as_date(sd)
        h = int(h)
        if h not in HORIZONS_TD:
            continue
        fwd = trading_days_in(t, t + dt.timedelta(days=FORWARD_CALENDAR_DAYS))
        if len(fwd) < h or any((u, d) not in present for d in fwd[:h]):
            coverage[(t.year, h)] += 1
            continue
        days = fwd[:h]
        end_date = days[-1]
        if corporate_action_in_window(corp.get(u, ()), splits.get(u, ()),
                                      t + dt.timedelta(days=1), end_date):
            dropped["corporate_action"] += 1
            continue
        down = sum(rs_down[(u, d)] for d in days)
        up = sum(rs_up[(u, d)] for d in days)
        implied = float(ftv)
        td, tu, reason = rules.compute_targets(down, up, implied)
        if reason is not None:
            dropped[reason] += 1
            continue
        weight = float(sup) if sup is not None else float("nan")
        out.append((u, t, h, implied, down, up, td, tu, weight))

    return pd.DataFrame(out, columns=list(_TARGET_COLUMNS)), dropped, coverage


def _as_date(x: Any) -> dt.date:
    return x.date() if isinstance(x, dt.datetime) else x
