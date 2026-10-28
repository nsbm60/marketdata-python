"""
Walk-forward check for the HAR forecasts (HAR edge spec §6) — the gate before §7 goes live.

From 2019 on, compare each stored forecast with the realized RV over its horizon:
  - realized  = mean rv_total over the next h sessions (from trading.rv_daily);
  - HAR       = sigma_forecast² / 252 (mean daily variance, from trading.rv_forecast);
  - baseline  = trailing 22-session mean rv_total on the forecast date (the naive predictor).
Losses: QLIKE = r − log r − 1 with r = realized/forecast, and squared error of log variance.

HAR must beat the trailing-22 baseline on QLIKE, pooled, at every horizon from 5 days up. The
report prints pooled and per-name losses and flags any horizon where it doesn't. It writes nothing.

    python tools/rv_eval.py [--since 2019-01-01]
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.shared.clickhouse import get_ch_client  # noqa: E402
from rv import HORIZONS, MODEL_VERSION  # noqa: E402
from rv import har  # noqa: E402


def _qlike(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    r = realized / forecast
    return r - np.log(r) - 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--model-version", default=MODEL_VERSION)
    args = ap.parse_args()
    since = dt.date.fromisoformat(args.since)

    ch = get_ch_client()

    # rv_daily series per name, once.
    rows = ch.query(
        "SELECT symbol, trading_date, rv_total FROM trading.rv_daily FINAL "
        "WHERE rv_total > 0 AND isFinite(rv_total) ORDER BY symbol, trading_date"
    ).result_rows
    series: dict[str, tuple[dict, dict, dict]] = {}
    acc: dict[str, list] = {}
    for sym, d, rv in rows:
        acc.setdefault(sym, [[], []])
        acc[sym][0].append(d)
        acc[sym][1].append(float(rv))

    # Per horizon, pooled loss accumulators and per-name means.
    pooled = {h: {"har_q": [], "base_q": [], "har_se": [], "base_se": []} for h in HORIZONS}
    per_name = {h: [] for h in HORIZONS}  # (symbol, n, har_q, base_q)

    for sym, (dates, vals) in acc.items():
        rv = np.asarray(vals, dtype=float)
        idx = {d: i for i, d in enumerate(dates)}
        trailing22 = har._trailing_mean(rv, 22)
        realized_lm = {h: har.forward_log_mean(rv, h) for h in HORIZONS}

        frows = ch.query(
            "SELECT as_of_date, horizon_days, sigma_forecast FROM trading.rv_forecast FINAL "
            f"WHERE model_version = '{args.model_version}' AND symbol = '{sym}' "
            f"AND as_of_date >= '{since.isoformat()}' ORDER BY as_of_date"
        ).result_rows

        name_loss = {h: {"har_q": [], "base_q": []} for h in HORIZONS}
        for as_of, h, sigma in frows:
            h = int(h)
            i = idx.get(as_of)
            if i is None:
                continue
            rlm = realized_lm[h][i]
            base = trailing22[i]
            if not (np.isfinite(rlm) and np.isfinite(base) and base > 0 and sigma > 0):
                continue
            realized = np.exp(rlm)
            har_var = (sigma * sigma) / 252.0
            har_q = float(_qlike(np.array([realized]), np.array([har_var]))[0])
            base_q = float(_qlike(np.array([realized]), np.array([base]))[0])
            har_se = (np.log(realized) - np.log(har_var)) ** 2
            base_se = (np.log(realized) - np.log(base)) ** 2
            pooled[h]["har_q"].append(har_q)
            pooled[h]["base_q"].append(base_q)
            pooled[h]["har_se"].append(float(har_se))
            pooled[h]["base_se"].append(float(base_se))
            name_loss[h]["har_q"].append(har_q)
            name_loss[h]["base_q"].append(base_q)
        for h in HORIZONS:
            q = name_loss[h]["har_q"]
            if q:
                per_name[h].append((sym, len(q), float(np.mean(q)), float(np.mean(name_loss[h]["base_q"]))))

    # Pooled report.
    print(f"HAR walk-forward — {args.model_version}, forecasts from {since} on")
    print(f"{'horizon':>7s} {'n':>9s} {'QLIKE_HAR':>10s} {'QLIKE_base':>11s} "
          f"{'logSE_HAR':>10s} {'logSE_base':>11s}  verdict")
    failures = []
    for h in HORIZONS:
        p = pooled[h]
        n = len(p["har_q"])
        if n == 0:
            # Nothing to score is a failure, not a pass — at any horizon.
            failures.append(h)
            print(f"{h:>7d} {0:>9d}  NO SCORED FORECASTS (gate fails)")
            continue
        hq, bq = float(np.mean(p["har_q"])), float(np.mean(p["base_q"]))
        hse, bse = float(np.mean(p["har_se"])), float(np.mean(p["base_se"]))
        beats = hq < bq
        verdict = "HAR wins" if beats else "HAR LOSES"
        if h >= 5 and not beats:
            failures.append(h)
        print(f"{h:>7d} {n:>9,d} {hq:>10.4f} {bq:>11.4f} {hse:>10.4f} {bse:>11.4f}  {verdict}")

    # Per-name: how many names HAR beats the baseline on, per horizon.
    print("\nper-name QLIKE (HAR vs baseline):")
    print(f"{'horizon':>7s} {'names':>6s} {'HAR_better':>11s} {'share':>7s}")
    for h in HORIZONS:
        names = per_name[h]
        if not names:
            continue
        better = sum(1 for _, _, hq, bq in names if hq < bq)
        print(f"{h:>7d} {len(names):>6d} {better:>11d} {better / len(names):>7.0%}")

    print()
    if failures:
        print(f"GATE FAILED at horizon(s): {', '.join(map(str, sorted(set(failures))))} "
              f"(HAR did not beat the trailing-22 baseline on pooled QLIKE, or nothing was scored).")
        sys.exit(1)
    print("GATE PASSED: HAR beats the trailing-22 baseline on pooled QLIKE at every horizon ≥ 5.")


if __name__ == "__main__":
    main()
