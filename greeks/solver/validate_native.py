"""python -m greeks.solver.validate_native <date> <underlying>

Read one (session_date, underlying) from trading.option_trade, price every row on BOTH paths — the
former pure-Python py_vollib path and the native "Let's Be Rational" path now in the core — and print
row counts, any status mismatch, and the maximum IV and greek differences. Writes nothing to the
database. Intended to be run on drogon against real data after a deploy.
"""

from __future__ import annotations

import math
import sys
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from py_vollib.black.greeks import analytical as _va
from py_vollib.black.implied_volatility import implied_volatility as _vollib_iv

from greeks.domain import FailureReason, OptionRight, RowStatus
from greeks.forwards.carry import carry
from greeks.occ import parse_occ
from greeks.solver.black76 import greeks_batch
from greeks.solver.invert import invert_batch

TABLE = "trading.option_trade"
_GREEKS = ("delta", "gamma", "vega", "theta", "rho")
_ABS_EPS = 1e-10


def _old_reason_and_iv(price: float, strike: float, is_call: bool, forward: float, discount: float,
                       t: float, *, t_floor_minutes: float) -> Tuple[object, float]:
    """The former scalar path: the pre-check ladder (first failing check wins), then py_vollib's IV.
    Returns (reason|None, iv) — iv is NaN on any failure."""
    flag = "c" if is_call else "p"
    if t < (t_floor_minutes * 60.0) / (365.0 * 86400.0):
        return FailureReason.T_BELOW_FLOOR, math.nan
    if not math.isfinite(price) or price < 0:
        return FailureReason.SOLVER_FAILED, math.nan
    if not math.isfinite(forward) or forward <= 0:
        return FailureReason.MISSING_FORWARD_INPUTS, math.nan
    if not math.isfinite(strike) or strike <= 0:
        return FailureReason.MISSING_FORWARD_INPUTS, math.nan
    if not math.isfinite(discount) or discount <= 0:
        return FailureReason.MISSING_FORWARD_INPUTS, math.nan
    intrinsic = (max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)) * discount
    upper = (forward if is_call else strike) * discount
    if price < intrinsic - _ABS_EPS:
        return FailureReason.BELOW_INTRINSIC, math.nan
    if price > upper + _ABS_EPS:
        return FailureReason.ABOVE_NO_ARB, math.nan
    if t < 1.0 / 365.0 and price <= intrinsic + 0.01 + _ABS_EPS:
        return FailureReason.QUANTIZED_NEAR_INTRINSIC, math.nan
    try:
        r = -math.log(discount) / t
        iv = float(_vollib_iv(price, forward, strike, r, t, flag))
    except Exception:  # noqa: BLE001
        return FailureReason.SOLVER_FAILED, math.nan
    if not math.isfinite(iv) or iv <= 0:
        return FailureReason.SOLVER_FAILED, math.nan
    return None, iv


def _old_greeks(flag: str, forward: float, strike: float, t: float, r: float, iv: float,
                spot: float) -> Dict[str, float]:
    d_fwd = float(_va.delta(flag, forward, strike, t, r, iv))
    g_fwd = float(_va.gamma(flag, forward, strike, t, r, iv))
    f_over_s = forward / spot
    return {
        "delta": d_fwd * f_over_s, "gamma": g_fwd * f_over_s * f_over_s,
        "vega": float(_va.vega(flag, forward, strike, t, r, iv)),
        "theta": float(_va.theta(flag, forward, strike, t, r, iv)),
        "rho": float(_va.rho(flag, forward, strike, t, r, iv)),
    }


def compare(forward: npt.NDArray[np.float64], strike: npt.NDArray[np.float64],
            is_call: npt.NDArray[np.bool_], discount: npt.NDArray[np.float64],
            tte: npt.NDArray[np.float64], price: npt.NDArray[np.float64],
            spot: npt.NDArray[np.float64], *, methodology_version: str,
            t_floor_minutes: float) -> Dict[str, Any]:
    """Pure: both paths over the staged (carry-computed) rows → the diff stats. No I/O."""
    n = int(price.shape[0])
    iv_new, status_new, reason_new = invert_batch(
        price, strike, is_call, forward, discount, tte,
        methodology_version=methodology_version, t_floor_minutes=t_floor_minutes)
    bg = greeks_batch(forward, strike, tte, discount,
                      np.where(np.isfinite(iv_new), iv_new, 1.0), is_call, spot)
    new_g = {"delta": bg.delta, "gamma": bg.gamma, "vega": bg.vega, "theta": bg.theta, "rho": bg.rho}

    status_mismatch = 0
    both_ok: List[int] = []
    iv_old = np.full(n, math.nan)
    g_old: Dict[str, npt.NDArray[np.float64]] = {k: np.full(n, math.nan) for k in _GREEKS}
    for i in range(n):
        rc, iv = _old_reason_and_iv(float(price[i]), float(strike[i]), bool(is_call[i]),
                                    float(forward[i]), float(discount[i]), float(tte[i]),
                                    t_floor_minutes=t_floor_minutes)
        new_ok = status_new[i] is RowStatus.SUCCESS
        if (rc is None) != new_ok or reason_new[i] != rc:
            status_mismatch += 1
            continue
        if rc is None:
            iv_old[i] = iv
            r = -math.log(float(discount[i])) / float(tte[i])
            gg = _old_greeks("c" if is_call[i] else "p", float(forward[i]), float(strike[i]),
                             float(tte[i]), r, iv, float(spot[i]))
            for k in _GREEKS:
                g_old[k][i] = gg[k]
            both_ok.append(i)

    ok = np.array(both_ok, dtype=int)
    stats: Dict[str, Any] = {"rows": n, "both_succeed": int(ok.size),
                             "status_mismatch": status_mismatch}
    if ok.size:
        stats["max_iv_rel"] = float(np.max(np.abs(iv_new[ok] - iv_old[ok]) / np.abs(iv_old[ok])))
        for k in _GREEKS:
            denom = np.maximum(np.abs(g_old[k][ok]), 1e-12)
            stats[f"max_{k}_rel"] = float(np.max(np.abs(new_g[k][ok] - g_old[k][ok]) / denom))
    return stats


def _read_pair(client: Any, day: date, underlying: str) -> List[Dict[str, Any]]:
    cols = "symbol, toFloat64(price) AS price, spot_at_trade, trade_ts, session_date"
    res = client.query(
        f"SELECT {cols} FROM {TABLE} FINAL WHERE underlying = %(u)s AND session_date = %(d)s",
        parameters={"u": underlying.upper(), "d": day})
    names = res.column_names
    return [dict(zip(names, row)) for row in res.result_rows]


def _as_utc(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _stage(rows: Sequence[Dict[str, Any]], inputs: Any, expiry_time_et: str) -> Dict[str, Any]:
    """Parse + carry each priceable row into arrays for compare(); count the pre-solver drops."""
    fwd: List[float] = []
    stk: List[float] = []
    call: List[bool] = []
    disc: List[float] = []
    tte: List[float] = []
    prc: List[float] = []
    spt: List[float] = []
    skipped = {"nonstandard": 0, "missing_spot": 0, "expired": 0, "carry_failed": 0}
    for row in rows:
        try:
            c = parse_occ(row["symbol"])
        except ValueError:
            skipped["nonstandard"] += 1
            continue
        spot = row["spot_at_trade"]
        if spot is None:
            skipped["missing_spot"] += 1
            continue
        if c.expiry < row["session_date"]:
            skipped["expired"] += 1
            continue
        try:
            cr = carry(float(spot), inputs.rate_for(c.expiry), _as_utc(row["trade_ts"]), c.expiry,
                       inputs.dividends_for(c.root), expiry_time_et=expiry_time_et)
        except Exception:  # noqa: BLE001
            skipped["carry_failed"] += 1
            continue
        fwd.append(cr.forward); stk.append(c.strike); call.append(c.right is OptionRight.CALL)
        disc.append(cr.discount); tte.append(cr.time_to_expiry); prc.append(float(row["price"]))
        spt.append(float(spot))
    return {"forward": np.array(fwd), "strike": np.array(stk), "is_call": np.array(call, dtype=bool),
            "discount": np.array(disc), "tte": np.array(tte), "price": np.array(prc),
            "spot": np.array(spt), "skipped": skipped}


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m greeks.solver.validate_native <YYYY-MM-DD> <underlying>", file=sys.stderr)
        return 2
    day = date.fromisoformat(argv[0])
    underlying = argv[1].upper()

    from greeks.ch import get_ch_client
    from greeks.config import load_config
    from greeks_server.inputs import clickhouse_loader
    cfg = load_config()
    client = get_ch_client()
    rows = _read_pair(client, day, underlying)
    inputs = clickhouse_loader(lambda: client)(day)
    staged = _stage(rows, inputs, cfg.expiry_time_et)
    stats = compare(staged["forward"], staged["strike"], staged["is_call"], staged["discount"],
                    staged["tte"], staged["price"], staged["spot"],
                    methodology_version=cfg.methodology_version, t_floor_minutes=float(cfg.t_floor_minutes))

    print(f"pair {day} {underlying}: {len(rows)} rows read; {int(staged['price'].shape[0])} reached the solver")
    print(f"  skipped before solver: {staged['skipped']}")
    print(f"  both paths succeed:    {stats['both_succeed']}")
    print(f"  status mismatches:     {stats['status_mismatch']}")
    if stats.get("both_succeed", 0):
        print(f"  max IV rel diff:       {stats['max_iv_rel']:.3e}")
        for k in _GREEKS:
            print(f"  max {k:<5} rel diff:   {stats[f'max_{k}_rel']:.3e}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
