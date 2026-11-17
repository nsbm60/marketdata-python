"""
Read-only inspection: `python -m iv_curve show <date> <underlying>` and
`python -m iv_curve coverage <from> <to>`. Neither writes anything.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, List, Optional, Tuple

from ml.shared.calendar import trading_days_between

from . import IV_CURVE_TABLE, KIND_EXPIRY, KIND_HORIZON, METHOD, STATUS_OK, TRADING_DAYS_PER_YEAR


def _fmt(x: Optional[float], width: int = 12, prec: int = 6) -> str:
    return f"{x:>{width}.{prec}f}" if x is not None else f"{'—':>{width}}"


def show(ch: Any, day: dt.date, underlying: str) -> None:
    """Print the curve for (day, underlying): expiry rows with the forward vol of each window from the
    previous point (the first from the anchor at (0,0)), then the horizon rows. Read-only."""
    res = ch.query(
        f"SELECT kind, tau_td, status, expiry, horizon_td, raw_total_var, fit_total_var, support, "
        f"bracket_lo_expiry, bracket_hi_expiry "
        f"FROM {IV_CURVE_TABLE} FINAL "
        "WHERE method = %(m)s AND underlying = %(u)s AND session_date = %(d)s "
        "ORDER BY kind, tau_td",
        parameters={"m": METHOD, "u": underlying.upper(), "d": day})

    print(f"iv_curve {METHOD}  {underlying.upper()}  session {day}")
    rows = res.result_rows
    if not rows:
        print("  (no rows)")
        return

    expiries = [r for r in rows if r[0] == KIND_EXPIRY]
    horizons = [r for r in rows if r[0] == KIND_HORIZON]

    print("  expiries (forward vol measured from the previous point; first from the anchor at τ=0):")
    print(f"    {'expiry':10s} {'τ':>4s} {'raw_var':>12s} {'fit_var':>12s} {'support':>10s} {'fwd_vol':>9s}")
    prev_tau = 0        # the anchor
    prev_fit = 0.0
    for _, tau_td, _status, expiry, _h, raw, fit, support, _lo, _hi in expiries:
        tau = int(tau_td)
        fwd = ""
        if fit is not None:
            dtau = tau - prev_tau
            dfit = float(fit) - prev_fit
            if dtau > 0:
                v = dfit / dtau * TRADING_DAYS_PER_YEAR  # per-trading-day forward variance, annualised
                fwd = f"{math.sqrt(v):>9.4f}" if v >= 0 else f"{'NEG':>9}"
            prev_tau, prev_fit = tau, float(fit)
        print(f"    {str(expiry):10s} {tau:>4d} {_fmt(raw)} {_fmt(fit)} "
              f"{_fmt(support, 10, 1)} {fwd:>9s}")

    print("  horizons:")
    print(f"    {'H':>4s} {'status':18s} {'fit_var':>12s} {'support':>10s}  brackets")
    for _, tau_td, status, _e, horizon_td, _raw, fit, support, lo, hi in horizons:
        bl = "anchor" if lo is None and status == STATUS_OK else (str(lo) if lo is not None else "—")
        bh = str(hi) if hi is not None else "—"
        print(f"    {int(horizon_td):>4d} {status:18s} {_fmt(fit)} {_fmt(support, 10, 1)}  {bl} → {bh}")


def coverage(ch: Any, start: dt.date, end: dt.date) -> None:
    """One line per underlying present in iv_curve_daily over [start, end]: trading days in range,
    % of days with an ok 21-day horizon point, and the median 21-day support — sorted by coverage
    descending. Read-only."""
    trading_days = trading_days_between(start - dt.timedelta(days=1), end)  # [start, end] inclusive
    res = ch.query(
        f"SELECT underlying, "
        f"  uniqExactIf(session_date, kind = %(kh)s AND horizon_td = 21 AND status = %(ok)s) AS ok_days, "
        f"  medianIf(support, kind = %(kh)s AND horizon_td = 21 AND status = %(ok)s) AS med_support "
        f"FROM {IV_CURVE_TABLE} FINAL "
        "WHERE method = %(m)s AND session_date BETWEEN %(a)s AND %(b)s "
        "GROUP BY underlying",
        parameters={"m": METHOD, "kh": KIND_HORIZON, "ok": STATUS_OK, "a": start, "b": end})

    rows: List[Tuple[str, int, float, Optional[float]]] = []
    for underlying, ok_days, med in res.result_rows:
        n = int(ok_days)
        pct = (100.0 * n / trading_days) if trading_days > 0 else 0.0
        rows.append((str(underlying), n, pct, (float(med) if n > 0 and med is not None else None)))
    rows.sort(key=lambda t: t[2], reverse=True)

    print(f"iv_curve {METHOD} coverage  {start} .. {end}  ({trading_days} trading days)")
    print(f"  {'underlying':12s} {'td':>5s} {'ok21%':>7s} {'med_support':>12s}")
    for underlying, _n, pct, med in rows:
        med_s = f"{med:>12.1f}" if med is not None else f"{'—':>12}"
        print(f"  {underlying:12s} {trading_days:>5d} {pct:>6.1f}% {med_s}")
