"""
Pooled HAR fit and the historical forecast backfill (HAR edge spec §4/§5).

Reads rv_total per name from trading.rv_daily, refits the pooled model weekly on the trailing four
years (only data available on the fit date), and writes to trading.rv_forecast the forecast HAR
would have made on every past session for each horizon — one row per (symbol, as_of_date, horizon).
A name with fewer than 66 complete sessions as of a date gets no forecast that date.

The numerics live in har.py (unit-tested); this module is the ClickHouse glue plus the weekly
rolling schedule and the per-name index windows.
"""

from __future__ import annotations

import bisect
import datetime as dt
import logging

import numpy as np

from . import HORIZONS, MIN_SESSIONS, MODEL_VERSION
from . import har

log = logging.getLogger("rv.forecast")

_FOUR_YEARS_DAYS = 4 * 365
_INSERT_COLS = [
    "symbol", "as_of_date", "horizon_days", "sigma_forecast",
    "model_version", "fit_date", "rv_d", "rv_w", "rv_m", "computed_at",
]


def load_series(ch) -> dict[str, tuple[list[dt.date], np.ndarray]]:
    """Per name, the ordered (dates, rv_total) over complete days with positive variance."""
    rows = ch.query(
        "SELECT symbol, trading_date, rv_total FROM trading.rv_daily FINAL "
        "WHERE rv_total > 0 ORDER BY symbol, trading_date"
    ).result_rows
    out: dict[str, tuple[list[dt.date], list[float]]] = {}
    for sym, d, rv in rows:
        dates, vals = out.setdefault(sym, ([], []))
        dates.append(d)
        vals.append(float(rv))
    return {s: (dates, np.asarray(vals, dtype=float)) for s, (dates, vals) in out.items()}


def weekly_fit_dates(all_dates: list[dt.date]) -> list[dt.date]:
    """The last trading day of each ISO week present — the weekly rolling refit schedule."""
    by_week: dict[tuple[int, int], dt.date] = {}
    for d in all_dates:
        iso = d.isocalendar()
        key = (iso[0], iso[1])
        if key not in by_week or d > by_week[key]:
            by_week[key] = d
    return sorted(by_week.values())


def _fit_models(
    preds: dict[str, np.ndarray],
    targets: dict[int, dict[str, np.ndarray]],
    dates_ord: dict[str, np.ndarray],
    fit_dates: list[dt.date],
) -> dict[dt.date, dict[int, tuple[np.ndarray, float]]]:
    """Pooled (beta, s²) per fit_date per horizon, using only data available on the fit date and
    within the trailing four years."""
    models: dict[dt.date, dict[int, tuple[np.ndarray, float]]] = {}
    for fd in fit_dates:
        fd_ord = fd.toordinal()
        lo_ord = fd_ord - _FOUR_YEARS_DAYS
        # per-name index window [window_start, last] inside the 4-year span, ending on/before fd
        last_idx, win_idx = {}, {}
        for sym, do in dates_ord.items():
            last = int(np.searchsorted(do, fd_ord, side="right")) - 1
            if last < 0:
                continue
            last_idx[sym] = last
            win_idx[sym] = int(np.searchsorted(do, lo_ord, side="left"))
        per_h: dict[int, tuple[np.ndarray, float]] = {}
        for h in HORIZONS:
            xs, ys = [], []
            tgt_h = targets[h]
            for sym, last in last_idx.items():
                hi = last - h  # inclusive: target's last session t+h <= last
                lo = win_idx[sym]
                if hi < lo:
                    continue
                xr = preds[sym][lo : hi + 1]
                yr = tgt_h[sym][lo : hi + 1]
                ok = np.isfinite(yr) & np.isfinite(xr).all(axis=1)
                if ok.any():
                    xs.append(xr[ok])
                    ys.append(yr[ok])
            if xs:
                x = np.vstack(xs)
                y = np.concatenate(ys)
                per_h[h] = har.fit_pooled(x, y)
        if per_h:
            models[fd] = per_h
    return models


def run(ch, batch_rows: int = 200_000) -> int:
    """Fit weekly and write the historical forecasts. Returns rows written."""
    series = load_series(ch)
    if not series:
        print("[forecast] no rv_daily rows — nothing to forecast")
        return 0

    all_dates = sorted({d for dates, _ in series.values() for d in dates})
    fit_dates = weekly_fit_dates(all_dates)
    fit_ords = [f.toordinal() for f in fit_dates]
    print(f"[forecast] {len(series)} names, {len(all_dates)} trading days, {len(fit_dates)} weekly fits")

    preds = {s: har.predictors(rv) for s, (_, rv) in series.items()}
    dates_ord = {s: np.array([d.toordinal() for d in dates]) for s, (dates, _) in series.items()}
    targets = {h: {s: har.forward_log_mean(rv, h) for s, (_, rv) in series.items()} for h in HORIZONS}

    models = _fit_models(preds, targets, dates_ord, fit_dates)
    if not models:
        print("[forecast] no fittable windows — nothing written")
        return 0
    model_fit_ords = sorted(m.toordinal() for m in models)

    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    buf: list[tuple] = []
    written = 0

    def flush():
        nonlocal written, buf
        if buf:
            ch.insert("trading.rv_forecast", buf, column_names=_INSERT_COLS)
            written += len(buf)
            buf = []

    for sym, (dates, rv) in series.items():
        pred = preds[sym]
        n = len(dates)
        for t in range(n):
            if t + 1 < MIN_SESSIONS:  # need >= 66 complete sessions up to and including today
                continue
            if not np.isfinite(pred[t]).all():
                continue
            as_of = dates[t]
            # most recent fit on or before today
            j = bisect.bisect_right(model_fit_ords, as_of.toordinal()) - 1
            if j < 0:
                continue
            fd = dt.date.fromordinal(model_fit_ords[j])
            per_h = models[fd]
            rv_d, rv_w, rv_m = float(pred[t, 0]), float(pred[t, 1]), float(pred[t, 2])
            for h in HORIZONS:
                mh = per_h.get(h)
                if mh is None:
                    continue
                beta, s2 = mh
                sigma = har.forecast_sigma(beta, s2, pred[t])
                buf.append((sym, as_of, int(h), sigma, MODEL_VERSION, fd, rv_d, rv_w, rv_m, now))
            if len(buf) >= batch_rows:
                flush()
    flush()
    print(f"[forecast] wrote {written:,} forecast rows ({MODEL_VERSION})")
    return written
