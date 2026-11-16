"""
The read-only term-structure inspection print: `python -m iv_term show <date> <underlying>`.

Reads trading.iv_term_daily for that pair and prints one line per expiry (expiry, days to expiry,
status, ATM IV, total variance, n_trades, n_strikes, k range), then the forward variance between
each pair of consecutive `ok` expiries, (V₂ − V₁)/(T₂ − T₁), annualised as a vol. A negative forward
variance is information about the data, not an error: its value is printed and marked NEGATIVE. This
writes nothing.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, List, Optional, Tuple

from . import IV_TERM_TABLE, METHOD, STATUS_OK


def _fmt(x: Optional[float], width: int = 9, prec: int = 4) -> str:
    return f"{x:>{width}.{prec}f}" if x is not None else f"{'—':>{width}}"


def show(ch: Any, day: dt.date, underlying: str) -> None:
    """Print the term structure for (day, underlying). Read-only."""
    res = ch.query(
        f"SELECT expiry, status, atm_iv, total_var, tte, n_trades, n_strikes, k_min, k_max "
        f"FROM {IV_TERM_TABLE} FINAL "
        "WHERE method = %(m)s AND underlying = %(u)s AND session_date = %(d)s "
        "ORDER BY expiry",
        parameters={"m": METHOD, "u": underlying.upper(), "d": day})

    print(f"iv_term {METHOD}  {underlying.upper()}  session {day}")
    if not res.result_rows:
        print("  (no rows)")
        return

    header = (f"  {'expiry':10s} {'dte':>4s} {'status':16s} {'atm_iv':>9s} {'total_var':>12s} "
              f"{'trades':>7s} {'strikes':>8s} {'k_min':>8s} {'k_max':>8s}")
    print(header)

    # (expiry, tte, total_var) for the consecutive-ok forward-variance pass.
    oks: List[Tuple[dt.date, float, float]] = []
    for expiry, status, atm_iv, total_var, tte, n_trades, n_strikes, k_min, k_max in res.result_rows:
        dte = (expiry - day).days
        print(f"  {str(expiry):10s} {dte:>4d} {status:16s} {_fmt(atm_iv)} {_fmt(total_var, 12, 6)} "
              f"{int(n_trades):>7d} {int(n_strikes):>8d} {_fmt(k_min, 8):>8s} {_fmt(k_max, 8):>8s}")
        if status == STATUS_OK and atm_iv is not None and total_var is not None and tte is not None:
            oks.append((expiry, float(tte), float(total_var)))

    if len(oks) < 2:
        return
    print("  forward variance between consecutive ok expiries (annualised vol):")
    for (e1, t1, v1), (e2, t2, v2) in zip(oks, oks[1:]):
        dt_years = t2 - t1
        if dt_years <= 0:
            # Non-increasing tte between two ok expiries — can't form a forward rate.
            print(f"  {e1} -> {e2}:  dT <= 0, skipped")
            continue
        fwd_var = (v2 - v1) / dt_years
        if fwd_var < 0:
            print(f"  {e1} -> {e2}:  fwd_var={fwd_var:+.6f}  NEGATIVE (total variance not monotone)")
        else:
            print(f"  {e1} -> {e2}:  fwd_var={fwd_var:.6f}  vol={math.sqrt(fwd_var):.4f}")
