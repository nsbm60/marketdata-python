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
import hashlib
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


def har_set_filter(ch) -> str:
    """` AND symbol IN (...)` for the latest trading.har_name_set snapshot, or '' (all names, with a
    warning) when the set hasn't been persisted yet. The Scala bar-load job writes it daily."""
    names = [r[0] for r in ch.query(
        "SELECT symbol FROM trading.har_name_set FINAL "
        "WHERE as_of_date = (SELECT max(as_of_date) FROM trading.har_name_set)"
    ).result_rows]
    if not names:
        log.warning("har_name_set is empty — covering ALL rv_daily names (run the bar-load job to persist the set)")
        return ""
    quoted = ", ".join("'" + s.replace("'", "") + "'" for s in names)
    return f" AND symbol IN ({quoted})"


def load_series(ch) -> dict[str, tuple[list[dt.date], np.ndarray]]:
    """Per name in the current HAR set, the ordered (dates, rv_total) over complete, positive days."""
    rows = ch.query(
        "SELECT symbol, trading_date, rv_total FROM trading.rv_daily FINAL "
        f"WHERE rv_total > 0 AND isFinite(rv_total){har_set_filter(ch)} ORDER BY symbol, trading_date"
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
                beta, s2 = har.fit_pooled(x, y)
                per_h[h] = (beta, s2, len(y))
        if per_h:
            models[fd] = per_h
    return models


def write_har_fit(ch, fit_date: dt.date, per_h: dict, now: dt.datetime) -> None:
    """Persist one weekly pooled fit's coefficients (one row per horizon) to trading.har_fit."""
    rows = [
        (MODEL_VERSION, fit_date, int(h), float(beta[0]), float(beta[1]), float(beta[2]),
         float(beta[3]), float(s2), int(n_obs), now)
        for h, (beta, s2, n_obs) in sorted(per_h.items())
    ]
    if rows:
        ch.insert("trading.har_fit", rows, column_names=[
            "model_version", "fit_date", "horizon_days", "b0", "b_d", "b_w", "b_m", "s2", "n_obs", "computed_at"])


def read_latest_har_fit(ch) -> dict:
    """Latest stored fit per horizon: {horizon: (beta[4], s2)} from trading.har_fit, newest fit_date."""
    rows = ch.query(
        "SELECT horizon_days, b0, b_d, b_w, b_m, s2 FROM trading.har_fit FINAL "
        f"WHERE model_version = '{MODEL_VERSION}' AND fit_date = "
        f"(SELECT max(fit_date) FROM trading.har_fit FINAL WHERE model_version = '{MODEL_VERSION}')"
    ).result_rows
    return {int(h): (np.array([b0, bd, bw, bm]), float(s2)) for h, b0, bd, bw, bm, s2 in rows}


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
    for fd, per_h in models.items():
        write_har_fit(ch, fd, per_h, now)
    print(f"[forecast] wrote {sum(len(v) for v in models.values())} har_fit rows across {len(models)} fits")
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
            # Store levels (not logs) to match the column names; pred holds logs, so exp them back.
            rv_d, rv_w, rv_m = float(np.exp(pred[t, 0])), float(np.exp(pred[t, 1])), float(np.exp(pred[t, 2]))
            for h in HORIZONS:
                mh = per_h.get(h)
                if mh is None:
                    continue
                beta, s2, _ = mh
                sigma = har.forecast_sigma(beta, s2, pred[t])
                buf.append((sym, as_of, int(h), sigma, MODEL_VERSION, fd, rv_d, rv_w, rv_m, now))
            if len(buf) >= batch_rows:
                flush()
    flush()
    print(f"[forecast] wrote {written:,} forecast rows ({MODEL_VERSION})")
    return written


def daily(ch) -> int:
    """The daily-job body (HAR edge §5): catch rv_daily up from the data, refit weekly into har_fit,
    and write each name's latest forecast from the latest stored fit. Invoked by the Scala HarJob.

    The rv_daily catch-up is incremental from the stored data, not the clock: each name is advanced
    from the day after its latest rv_daily row up to the latest complete ledger day (rv_daily.py),
    so a late run computes the day anyway and a missed day is caught up. Forecasts are written as of
    each name's latest rv_daily day."""
    from . import rv_daily

    rv_daily.incremental(ch)

    series = load_series(ch)
    if not series:
        print("[daily] no rv_daily rows")
        return 0
    all_dates = sorted({d for dates, _ in series.values() for d in dates})
    latest = all_dates[-1]
    preds = {s: har.predictors(rv) for s, (_, rv) in series.items()}
    dates_ord = {s: np.array([d.toordinal() for d in dates]) for s, (dates, _) in series.items()}
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

    # Weekly refit: fit as of the latest complete day when there's no stored fit or it's ≥ 7 days old.
    last_fit = ch.query(
        f"SELECT max(fit_date) FROM trading.har_fit FINAL WHERE model_version = '{MODEL_VERSION}'"
    ).result_rows[0][0]
    if last_fit is None or (latest - last_fit).days >= 7:
        targets = {h: {s: har.forward_log_mean(rv, h) for s, (_, rv) in series.items()} for h in HORIZONS}
        per_h = _fit_models(preds, targets, dates_ord, [latest]).get(latest, {})
        if per_h:
            write_har_fit(ch, latest, per_h, now)
            print(f"[daily] refit as of {latest}: {len(per_h)} horizons")

    models = read_latest_har_fit(ch)
    fit_date = ch.query(
        f"SELECT max(fit_date) FROM trading.har_fit FINAL WHERE model_version = '{MODEL_VERSION}'"
    ).result_rows[0][0]
    if not models:
        print("[daily] no stored fit — nothing to forecast")
        return 0

    buf: list[tuple] = []
    for sym, (dates, rv) in series.items():
        if len(dates) < MIN_SESSIONS:
            continue
        t = len(dates) - 1
        pred = preds[sym]
        if not np.isfinite(pred[t]).all():
            continue
        as_of = dates[t]  # each name's latest complete day
        rv_d, rv_w, rv_m = float(np.exp(pred[t, 0])), float(np.exp(pred[t, 1])), float(np.exp(pred[t, 2]))
        for h in HORIZONS:
            mh = models.get(h)
            if mh is None:
                continue
            beta, s2 = mh
            sigma = har.forecast_sigma(beta, s2, pred[t])
            buf.append((sym, as_of, int(h), sigma, MODEL_VERSION, fit_date, rv_d, rv_w, rv_m, now))
    if buf:
        ch.insert("trading.rv_forecast", buf, column_names=_INSERT_COLS)
    print(f"[daily] wrote {len(buf)} forecast rows (as of each name's latest complete day)")

    _sync_no_forecast_attention(ch, forecast_syms={row[0] for row in buf}, series=series)
    return len(buf)


# ── attention items for har_name_set names that produced no forecast (surfaced, not only logged) ──

def _har_name_set(ch) -> set[str]:
    return {r[0] for r in ch.query(
        "SELECT symbol FROM trading.har_name_set FINAL "
        "WHERE as_of_date = (SELECT max(as_of_date) FROM trading.har_name_set)").result_rows}


def _attention_raise(ch, kind: str, subject: str, symbol: str | None, detail: str) -> None:
    """Mirror Scala AttentionItemRepository: item_id = sha1(kind|subject); never reopen a resolved
    item; otherwise upsert an open row (now64 server-side, so the UTC epoch is correct)."""
    item_id = hashlib.sha1(f"{kind}|{subject}".encode("utf-8")).hexdigest()
    resolved = ch.query(
        f"SELECT count() FROM trading.attention_item FINAL "
        f"WHERE item_id='{item_id}' AND status='resolved'").result_rows[0][0]
    if resolved:
        return
    symv = "NULL" if symbol is None else f"'{symbol}'"
    ch.command(
        "INSERT INTO trading.attention_item (item_id,kind,symbol,detail,raised_at,status,updated_at) "
        f"VALUES ('{item_id}','{kind}',{symv},'{detail.replace(chr(39), chr(39) * 2)}', now64(3),'open', now64(3))")


def _attention_resolve(ch, kind: str, subject: str, symbol: str | None) -> None:
    """Resolve an open item (preserving its raised_at), e.g. a no_forecast item for a name that now
    forecasts. No-op if there is no open row."""
    item_id = hashlib.sha1(f"{kind}|{subject}".encode("utf-8")).hexdigest()
    rows = ch.query(
        f"SELECT detail, toUnixTimestamp64Milli(raised_at) FROM trading.attention_item FINAL "
        f"WHERE item_id='{item_id}' AND status='open'").result_rows
    if not rows:
        return
    detail, raised_ms = rows[0]
    symv = "NULL" if symbol is None else f"'{symbol}'"
    ch.command(
        "INSERT INTO trading.attention_item (item_id,kind,symbol,detail,raised_at,status,updated_at) "
        f"VALUES ('{item_id}','{kind}',{symv},'{detail.replace(chr(39), chr(39) * 2)}', "
        f"fromUnixTimestamp64Milli({raised_ms}),'resolved', now64(3))")


def _sync_no_forecast_attention(ch, forecast_syms: set[str], series: dict) -> None:
    """A har_name_set name with no forecast is a data problem worth surfacing, not just logging. Raise
    a `no_forecast` attention item for each such name (with its rv-session count — the §5 >=66 rule),
    and resolve the item for any name that now forecasts again."""
    try:
        har = _har_name_set(ch)
        sessions = {s: len(dates) for s, (dates, _) in series.items()}
        missing = sorted(har - forecast_syms)
        for sym in missing:
            n = sessions.get(sym, 0)
            detail = (f"{sym} is in har_name_set but has no rv_forecast — "
                      + (f"only {n} rv sessions (< {MIN_SESSIONS} required)" if n > 0
                         else "no rv_daily history (needs an rv backfill)"))
            _attention_raise(ch, "no_forecast", f"no_forecast|{sym}", sym, detail)
        # Self-clear: resolve the no_forecast item for any name that forecasts again.
        open_nf = {r[0] for r in ch.query(
            "SELECT symbol FROM trading.attention_item FINAL "
            "WHERE kind='no_forecast' AND status='open'").result_rows}
        for sym in open_nf & forecast_syms:
            _attention_resolve(ch, "no_forecast", f"no_forecast|{sym}", sym)
        if missing:
            print(f"[daily] {len(missing)} har_name_set names have no forecast "
                  f"(raised no_forecast attention items): {missing[:20]}")
    except Exception as exc:  # noqa: BLE001 — attention bookkeeping must never fail the forecast job
        log.warning("could not sync no_forecast attention items: %s", exc)
