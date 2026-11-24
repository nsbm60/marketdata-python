"""
The one input builder, shared by training and the later live forecast.

build_inputs(ch, start, end) reads ClickHouse into DataFrames and hands them to the pure
assemble_inputs(frames, start, end), which produces the per-(underlying, t, H) inputs using ONLY data
dated at or before each t's close. The dataset `build` calls this month by month; the live forecast
will call it for a single day. There is no second path.

Look-ahead safety: every trailing statistic is a backward rolling window, and the curve / event /
earnings-history / own-earnings-outcome reads are point-in-time at or before t. The one deliberate
exception is the CONFIRMED earnings date: the vendor publishes a confirmed report date up to
EARNINGS_CONFIRMED_DAYS ahead, so a report dated within that window of t is treated as known at t
(documented below and in the look-ahead test, which leaves that one row fixed).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from ml.shared.calendar import next_trading_day, trading_days_between

from . import (
    CURVE_KIND_HORIZON, CURVE_METHOD, CURVE_STATUS_OK, EARNINGS_HISTORY_N, EARNINGS_HISTORY_TABLE,
    HORIZONS_TD, IV_CURVE_TABLE, IV_EVENT_OUTCOME_TABLE, IV_EVENT_TABLE, JUMP_DAILY_TABLE,
    JUMP_METHOD, LOOKBACK_CALENDAR_DAYS, MEAN_WINDOWS, OUTCOME_METHOD, OUTCOME_OWN_EARNINGS,
    REGIME_WINDOW, RV_DAILY_TABLE, RV_SOURCE, SPY_SYMBOL, SURPRISE_WINDOW, UNSPLIT_WINDOW,
    WINDOW_METHOD, WINDOW_STATUS_OK,
)
from . import rules

# Column groups, frozen here and recorded in the manifest.
_MEAN_COLS = ("c", "j", "rs_down", "rs_up")   # the realized series that get trailing means and scaling


# ── ClickHouse reads (each returns a DataFrame; no logic beyond typing) ────────────────────────────

@dataclass(frozen=True)
class Frames:
    """Every source the inputs need, as DataFrames (plus the market-wide date set). assemble_inputs is
    a pure function of these — the look-ahead test perturbs future rows here and asserts t is stable."""

    rv: pd.DataFrame            # symbol, trading_date, rv_total, rv_overnight, rs_down, rs_up
    jump: pd.DataFrame          # symbol, trading_date, intraday_jump, jump_var, overnight_jump, overnight_jump_adj
    curve: pd.DataFrame         # underlying, session_date, horizon_td, status, fit_total_var, support
    events: pd.DataFrame        # underlying, session_date, end_td, eff_days, start_td, excess_days
    earnings: pd.DataFrame      # symbol, reported_date, reaction_date, vendor_report_time
    outcomes: pd.DataFrame      # underlying, end_expiry, realized_excess_var, excess_var, normal_daily_var
    market_wide: Set[dt.date]   # dates meeting iv_event's market-wide rule


def _q(ch: Any, sql: str, params: Dict[str, Any]) -> List[tuple]:
    return ch.query(sql, parameters=params).result_rows


def _read_frames(ch: Any, start: dt.date, end: dt.date) -> Frames:
    """Read every source for building inputs over [start, end]. rv/jump cover a trailing lookback (for
    the rolling windows); curve/events only the as-of month; earnings/outcomes are read per name
    (small). The universe is the names with a curve horizon row in the month, plus SPY (for regime)."""
    lo = start - dt.timedelta(days=LOOKBACK_CALENDAR_DAYS)

    curve = pd.DataFrame(_q(
        ch,
        f"SELECT underlying, session_date, horizon_td, status, fit_total_var, support "
        f"FROM {IV_CURVE_TABLE} FINAL "
        f"WHERE method = %(m)s AND kind = %(k)s AND session_date BETWEEN %(a)s AND %(b)s",
        {"m": CURVE_METHOD, "k": CURVE_KIND_HORIZON, "a": start, "b": end}),
        columns=["underlying", "session_date", "horizon_td", "status", "fit_total_var", "support"])

    names = {str(u).upper() for u in curve["underlying"].unique()} | {SPY_SYMBOL}
    names_l = sorted(names)

    rv = pd.DataFrame(_q(
        ch,
        f"SELECT symbol, trading_date, rv_total, rv_overnight, rs_down, rs_up "
        f"FROM {RV_DAILY_TABLE} FINAL "
        f"WHERE source = %(s)s AND symbol IN %(u)s AND trading_date BETWEEN %(a)s AND %(b)s",
        {"s": RV_SOURCE, "u": names_l, "a": lo, "b": end}),
        columns=["symbol", "trading_date", "rv_total", "rv_overnight", "rs_down", "rs_up"])

    jump = pd.DataFrame(_q(
        ch,
        f"SELECT symbol, trading_date, intraday_jump, jump_var, overnight_jump, overnight_jump_adj "
        f"FROM {JUMP_DAILY_TABLE} FINAL "
        f"WHERE method = %(m)s AND symbol IN %(u)s AND trading_date BETWEEN %(a)s AND %(b)s",
        {"m": JUMP_METHOD, "u": names_l, "a": lo, "b": end}),
        columns=["symbol", "trading_date", "intraday_jump", "jump_var", "overnight_jump",
                 "overnight_jump_adj"])

    events = pd.DataFrame(_q(
        ch,
        f"SELECT underlying, session_date, end_td, eff_days, start_td, excess_days "
        f"FROM {IV_EVENT_TABLE} FINAL "
        f"WHERE method = %(m)s AND status = %(s)s AND session_date BETWEEN %(a)s AND %(b)s",
        {"m": WINDOW_METHOD, "s": WINDOW_STATUS_OK, "a": start, "b": end}),
        columns=["underlying", "session_date", "end_td", "eff_days", "start_td", "excess_days"])

    earnings = pd.DataFrame(_q(
        ch,
        f"SELECT symbol, reported_date, reaction_date, vendor_report_time "
        f"FROM {EARNINGS_HISTORY_TABLE} FINAL WHERE symbol IN %(u)s",
        {"u": names_l}),
        columns=["symbol", "reported_date", "reaction_date", "vendor_report_time"])

    # Own-earnings outcomes joined to their window (for excess_var / normal_daily_var); end_expiry only
    # up to the month end — the per-t filter (end_expiry < t) is applied in assemble_inputs.
    outcomes = pd.DataFrame(_q(
        ch,
        f"SELECT o.underlying, o.end_expiry, o.realized_excess_var, w.excess_var, w.normal_daily_var "
        f"FROM {IV_EVENT_OUTCOME_TABLE} AS o FINAL "
        f"INNER JOIN {IV_EVENT_TABLE} AS w FINAL "
        f"  ON w.underlying = o.underlying AND w.session_date = o.session_date AND w.end_td = o.end_td "
        f"WHERE o.method = %(om)s AND o.explained = %(e)s AND o.underlying IN %(u)s "
        f"AND o.end_expiry <= %(b)s AND w.method = %(wm)s AND w.status = %(ws)s",
        {"om": OUTCOME_METHOD, "e": OUTCOME_OWN_EARNINGS, "u": names_l, "b": end,
         "wm": WINDOW_METHOD, "ws": WINDOW_STATUS_OK}),
        columns=["underlying", "end_expiry", "realized_excess_var", "excess_var", "normal_daily_var"])

    from iv_event.outcome_event import read_market_wide
    market_wide = read_market_wide(ch, lo, end)

    return Frames(rv=rv, jump=jump, curve=curve, events=events, earnings=earnings,
                  outcomes=outcomes, market_wide=market_wide)


def build_inputs(ch: Any, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Per-(underlying, t, H) inputs for every (underlying, t) with an ok curve horizon row, t in
    [start, end]. Look-ahead-safe. Live and training both call this."""
    return assemble_inputs(_read_frames(ch, start, end), start, end)


# ── the pure assembly (look-ahead-safe; no ClickHouse) ─────────────────────────────────────────────

def _as_date(x: Any) -> Optional[dt.date]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, pd.Timestamp):
        return x.date()
    return x


def _daily(rv: pd.DataFrame, jump: pd.DataFrame, market_wide: Set[dt.date],
           own_reaction: Set[Tuple[str, dt.date]]) -> pd.DataFrame:
    """One row per (symbol, trading_date): the C/J split, the realized series, the trailing means, and
    the surprise statistics — every column a backward rolling window, so t never sees a later day."""
    d = rv.merge(jump, on=["symbol", "trading_date"], how="left")
    d["symbol"] = d["symbol"].astype(str).str.upper()
    d = d.sort_values(["symbol", "trading_date"]).reset_index(drop=True)

    # The vectorised C/J split — identical semantics to rules.cj_split (which the tests pin), done
    # columnwise here for speed over the trailing window.
    ij, oj, oja = d["intraday_jump"], d["overnight_jump"], d["overnight_jump_adj"]
    # A flagged day with a NULL jump_var contradicts jump_daily's definition — fail, never fill.
    flagged_null = (ij == 1) & d["jump_var"].isna()
    if bool(flagged_null.any()):
        bad = d.loc[flagged_null, ["symbol", "trading_date"]].iloc[0]
        raise ValueError(
            f"jump_daily {bad['symbol']} {bad['trading_date']}: intraday_jump = 1 but jump_var is "
            "NULL — contradicts jump_daily's definition (jump_var is set on a flagged day)")
    intraday_part = np.where(ij == 1, d["jump_var"], 0.0)
    overnight_part = np.where((oj == 1) | (oja == 1), d["rv_overnight"], 0.0)
    d["j"] = intraday_part + overnight_part
    d["c"] = d["rv_total"] - d["j"]
    d["unsplit"] = ij.isna().astype(int)
    d["is_jump"] = ((ij == 1) | (oj == 1) | (oja == 1)).astype(int)

    g = d.groupby("symbol", sort=False)
    for col in _MEAN_COLS:
        for w in MEAN_WINDOWS:
            d[f"{col}_mean_{w}"] = (
                g[col].rolling(w, min_periods=w).mean().reset_index(level=0, drop=True))
    # Same lookback rule as the other 22-day inputs: NULL with fewer than UNSPLIT_WINDOW sessions.
    d["n_unsplit_days_22"] = (
        g["unsplit"].rolling(UNSPLIT_WINDOW, min_periods=UNSPLIT_WINDOW).sum()
        .reset_index(level=0, drop=True))
    # Each day's own trailing-22 continuous mean, for the surprise jump ratio J_d / C_22.
    c22 = g["c"].rolling(UNSPLIT_WINDOW, min_periods=UNSPLIT_WINDOW).mean().reset_index(level=0, drop=True)

    idx = pd.MultiIndex.from_arrays([d["symbol"], d["trading_date"]])
    is_own = idx.isin(own_reaction)
    is_mw = d["trading_date"].isin(market_wide).to_numpy()
    is_surprise = (d["is_jump"] == 1).to_numpy() & ~is_own & ~is_mw
    ratio = np.where(c22 > 0, d["j"] / c22, np.nan)
    valid = is_surprise & ~np.isnan(ratio)   # a surprise day with a usable trailing-22 C mean
    d["_surprise"] = is_surprise.astype(int)
    # The contribution is 0 (not NaN) on non-surprise / no-C_22 days so the trailing-252 rolling sum is
    # not killed by its min_periods (a mostly-NaN column never reaches 252 non-NaN observations — the
    # bug that left surprise_mean_jump_ratio NULL on every row); _ratio_n carries the real denominator.
    d["_ratio_contrib"] = np.where(valid, ratio, 0.0)
    d["_ratio_n"] = valid.astype(int)

    gg = d.groupby("symbol", sort=False)
    scount = gg["_surprise"].rolling(SURPRISE_WINDOW, min_periods=SURPRISE_WINDOW).sum()
    rsum = gg["_ratio_contrib"].rolling(SURPRISE_WINDOW, min_periods=SURPRISE_WINDOW).sum()
    rn = gg["_ratio_n"].rolling(SURPRISE_WINDOW, min_periods=SURPRISE_WINDOW).sum()
    d["surprise_count_252"] = scount.reset_index(level=0, drop=True)
    rsum = rsum.reset_index(level=0, drop=True)
    rn = rn.reset_index(level=0, drop=True)
    d["surprise_mean_jump_ratio"] = np.where(rn > 0, rsum / rn, np.nan)
    return d


def _curve_by_pair(curve: pd.DataFrame) -> Dict[Tuple[str, dt.date], dict]:
    """(underlying, session_date) -> {iv_perday[H], support[H], ok:set(H)} from the ok horizon rows."""
    out: Dict[Tuple[str, dt.date], dict] = {}
    for u, sd, h, status, ftv, sup in curve.itertuples(index=False):
        key = (str(u).upper(), _as_date(sd))
        rec = out.setdefault(key, {"iv": {}, "support": {}, "ok": set()})
        h = int(h)
        if h not in HORIZONS_TD:
            continue
        if status == CURVE_STATUS_OK and ftv is not None and not pd.isna(ftv):
            rec["iv"][h] = float(ftv) / h
            rec["support"][h] = None if sup is None or pd.isna(sup) else float(sup)
            rec["ok"].add(h)
    return out


def _events_by_pair(events: pd.DataFrame) -> Dict[Tuple[str, dt.date], List[tuple]]:
    out: Dict[Tuple[str, dt.date], List[tuple]] = {}
    for u, sd, end_td, eff, start_td, exc in events.itertuples(index=False):
        key = (str(u).upper(), _as_date(sd))
        out.setdefault(key, []).append((int(end_td), float(eff), int(start_td), float(exc)))
    return out


def _earnings_by_symbol(earnings: pd.DataFrame
                        ) -> Tuple[Dict[str, List[Tuple[dt.date, Optional[str]]]], Set[Tuple[str, dt.date]]]:
    """(reports per symbol as sorted (reported_date, vendor_time); the set of (symbol, own-reaction
    day) where the reaction day is coalesce(reaction_date, reported_date))."""
    reports: Dict[str, List[Tuple[dt.date, Optional[str]]]] = {}
    own: Set[Tuple[str, dt.date]] = set()
    for sym, reported, reaction, vtime in earnings.itertuples(index=False):
        s = str(sym).upper()
        rd = _as_date(reported)
        if rd is None:
            continue
        reports.setdefault(s, []).append((rd, None if vtime is None else str(vtime)))
        rx = _as_date(reaction)
        own.add((s, rx if rx is not None else rd))
    for s in reports:
        reports[s].sort(key=lambda r: r[0])
    return reports, own


def _outcomes_by_underlying(outcomes: pd.DataFrame) -> Dict[str, List[tuple]]:
    """underlying -> (end_expiry, realized_over_priced, realized_excess_days) sorted by end_expiry."""
    out: Dict[str, List[tuple]] = {}
    for u, ee, rxv, xv, ndv in outcomes.itertuples(index=False):
        ee = _as_date(ee)
        if ee is None or rxv is None or pd.isna(rxv):
            continue
        rop = (float(rxv) / float(xv)) if (xv not in (None, 0) and not pd.isna(xv)) else None
        red = (float(rxv) / float(ndv)) if (ndv not in (None, 0) and not pd.isna(ndv)) else None
        out.setdefault(str(u).upper(), []).append((ee, rop, red))
    for u in out:
        out[u].sort(key=lambda r: r[0])
    return out


def _earnings_history_medians(rows: List[tuple], t: dt.date) -> Tuple[Optional[float], Optional[float]]:
    """Medians of realized/priced and realized excess days over the last EARNINGS_HISTORY_N own-earnings
    outcomes with end_expiry < t."""
    past = [r for r in rows if r[0] < t][-EARNINGS_HISTORY_N:]
    rop = [r[1] for r in past if r[1] is not None]
    red = [r[2] for r in past if r[2] is not None]
    return (float(np.median(rop)) if rop else None, float(np.median(red)) if red else None)


def _regime_by_date(daily: pd.DataFrame, curve_pairs: Dict[Tuple[str, dt.date], dict],
                    market_wide: Set[dt.date]) -> Dict[dt.date, dict]:
    """SPY's regime per date: per-day implied var at 21, log(iv42/iv5), C/J 22-day means, and the count
    of market-wide dates in the last REGIME_WINDOW sessions (SPY's session axis)."""
    spy = daily[daily["symbol"] == SPY_SYMBOL].sort_values("trading_date").reset_index(drop=True)
    mw_flag = spy["trading_date"].isin(market_wide).astype(int)
    mw_count = mw_flag.rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW).sum()
    out: Dict[dt.date, dict] = {}
    for i, row in spy.iterrows():
        d = _as_date(row["trading_date"])
        cu = curve_pairs.get((SPY_SYMBOL, d), {"iv": {}})
        iv = cu["iv"]
        slope = (np.log(iv[42] / iv[5]) if (5 in iv and 42 in iv and iv[5] > 0) else np.nan)
        out[d] = {
            "spy_iv_perday_21": iv.get(21, np.nan),
            "spy_iv_slope_42_5": slope,
            "spy_c_mean_22": row.get("c_mean_22", np.nan),
            "spy_j_mean_22": row.get("j_mean_22", np.nan),
            "market_wide_count_22": mw_count.iloc[i],
        }
    return out


def assemble_inputs(frames: Frames, start: dt.date, end: dt.date) -> pd.DataFrame:
    """The pure, look-ahead-safe assembly: per (underlying, t, H) inputs for every (underlying, t) with
    an ok curve horizon row and t in [start, end]."""
    reports, own_reaction = _earnings_by_symbol(frames.earnings)
    daily = _daily(frames.rv, frames.jump, frames.market_wide, own_reaction)
    daily_by_pair = {(r.symbol, _as_date(r.trading_date)): r
                     for r in daily.itertuples(index=False)}
    curve_pairs = _curve_by_pair(frames.curve)
    events_pairs = _events_by_pair(frames.events)
    outcomes = _outcomes_by_underlying(frames.outcomes)
    regime = _regime_by_date(daily, curve_pairs, frames.market_wide)

    rows: List[dict] = []
    for (u, t), cu in sorted(curve_pairs.items()):
        if not (start <= t <= end) or not cu["ok"]:
            continue
        drow = daily_by_pair.get((u, t))
        base = _base_row(u, t, cu, drow, reports.get(u, []), outcomes.get(u, []), regime.get(t))
        ev = events_pairs.get((u, t), [])
        for h in sorted(cu["ok"]):
            rows.append(_expand_h(base, cu, ev, h))
    df = pd.DataFrame(rows, columns=_COLUMNS)
    # n_unsplit_days_22 is a nullable count (NULL when fewer than UNSPLIT_WINDOW sessions, or no rv
    # row at t): a true nullable integer, never 0 as a stand-in for unknown.
    df["n_unsplit_days_22"] = df["n_unsplit_days_22"].astype("Int64")
    return df


def _base_row(u: str, t: dt.date, cu: dict, drow: Any, name_reports: Sequence[tuple],
              name_outcomes: List[tuple], reg: Optional[dict]) -> dict:
    """The per-(underlying, t) fields (same across H): realized means, curve term structure, the
    earnings view, surprise and regime. drow is the daily row at t (None if the name has no rv row)."""
    iv = cu["iv"]
    sup = cu["support"]
    row: dict = {"underlying": u, "t": t}

    for col in _MEAN_COLS:
        for w in MEAN_WINDOWS:
            row[f"{col}_mean_{w}"] = _get(drow, f"{col}_mean_{w}")
    row["n_unsplit_days_22"] = _nullable_int(_get(drow, "n_unsplit_days_22"))
    row["surprise_count_252"] = _get(drow, "surprise_count_252")
    row["surprise_mean_jump_ratio"] = _get(drow, "surprise_mean_jump_ratio")

    for h in HORIZONS_TD:
        row[f"iv_perday_{h}"] = iv.get(h, np.nan)
        row[f"iv_support_{h}"] = sup.get(h, np.nan)
    row["iv_slope_42_5"] = (np.log(iv[42] / iv[5]) if (5 in iv and 42 in iv and iv[5] > 0) else np.nan)
    row["iv_slope_21_5"] = (np.log(iv[21] / iv[5]) if (5 in iv and 21 in iv and iv[5] > 0) else np.nan)

    e = rules.earnings_as_of(t, name_reports, next_trading_day, trading_days_between)
    row["earnings_date"] = e.earnings_date
    row["earnings_date_status"] = e.status
    row["report_time"] = e.report_time
    row["earnings_reaction_session"] = e.reaction_session
    row["earnings_td"] = (None if e.earnings_td is None else int(e.earnings_td))
    rop, red = _earnings_history_medians(name_outcomes, t)
    row["earnings_hist_median_realized_over_priced"] = rop
    row["earnings_hist_median_realized_excess_days"] = red

    reg = reg or {}
    for k in ("spy_iv_perday_21", "spy_iv_slope_42_5", "spy_c_mean_22", "spy_j_mean_22",
              "market_wide_count_22"):
        row[k] = reg.get(k, np.nan)
    return row


def _expand_h(base: dict, cu: dict, events: List[tuple], h: int) -> dict:
    """One output row: the per-(underlying, t) base plus the H-specific fields — the scaled means, the
    priced-event sums over end_td <= H, and whether the earnings reaction falls within H.

    Scaling puts each realized mean on the target's per-day-implied-variance scale (iv_perday at H).
    The jump means use a PLAIN ratio (j_mean / iv_perday): 0 jumps is a real 0, not missing, so a log
    would drop most days. The C, rs_down and rs_up means use a log ratio (log(mean / iv_perday)); each
    is NULL only when its mean or the implied value is NULL (and, for the logs, when the mean is 0)."""
    row = dict(base)
    row["H"] = h
    iv_h = cu["iv"].get(h, np.nan)
    iv_ok = (not _isnan(iv_h)) and iv_h > 0
    for col in _MEAN_COLS:
        for w in MEAN_WINDOWS:
            m = base[f"{col}_mean_{w}"]
            key = f"{col}_mean_{w}_scaled"
            if m is None or _isnan(m) or not iv_ok:
                row[key] = np.nan
            elif col == "j":
                row[key] = float(m / iv_h)              # plain ratio — 0 jumps is a real 0
            elif m > 0:
                row[key] = float(np.log(m / iv_h))      # C / rs_down / rs_up stay logs (need m > 0)
            else:
                row[key] = np.nan
    rule_n, rule_exc, other_exc = rules.event_window_sums(events, h)
    row["event_rule_count"] = rule_n
    row["event_rule_excess_days"] = rule_exc
    row["event_other_excess_days"] = other_exc
    td = base["earnings_td"]
    row["earnings_within_h"] = (td is not None and td <= h)
    return row


def _get(drow: Any, attr: str) -> Any:
    if drow is None:
        return np.nan
    v = getattr(drow, attr, np.nan)
    return np.nan if (v is None) else v


def _isnan(x: Any) -> bool:
    return isinstance(x, float) and np.isnan(x)


def _nullable_int(x: Any) -> Any:
    """An integer, or pd.NA when the value is unknown (NaN/None) — never 0 as a stand-in."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NA
    return int(round(x))


# The frozen output column order (recorded in the manifest).
_COLUMNS: Tuple[str, ...] = (
    "underlying", "t", "H",
    *(f"{c}_mean_{w}" for c in _MEAN_COLS for w in MEAN_WINDOWS),
    *(f"{c}_mean_{w}_scaled" for c in _MEAN_COLS for w in MEAN_WINDOWS),
    "n_unsplit_days_22",
    *(f"iv_perday_{h}" for h in HORIZONS_TD),
    *(f"iv_support_{h}" for h in HORIZONS_TD),
    "iv_slope_42_5", "iv_slope_21_5",
    "event_rule_count", "event_rule_excess_days", "event_other_excess_days",
    "earnings_date", "earnings_date_status", "report_time", "earnings_reaction_session",
    "earnings_td", "earnings_within_h",
    "earnings_hist_median_realized_over_priced", "earnings_hist_median_realized_excess_days",
    "surprise_count_252", "surprise_mean_jump_ratio",
    "spy_iv_perday_21", "spy_iv_slope_42_5", "spy_c_mean_22", "spy_j_mean_22",
    "market_wide_count_22",
)
