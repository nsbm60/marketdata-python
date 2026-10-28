"""
The HAR model (HAR edge spec §4) — pure numpy, no ClickHouse.

For each horizon h, pooled across names:

    log RV(t -> t+h) = b0 + b_d·log RV_d(t) + b_w·log RV_w(t) + b_m·log RV_m(t)

  RV_d(t) = rv_total(t); RV_w(t) = mean rv_total over the last 5 sessions; RV_m(t) over the last 22.
  RV(t -> t+h) = mean rv_total over the next h sessions (t+1 … t+h).
  All regressors and the target are logs of averaged *levels* (log of the mean, not mean of logs).

The log form forecasts the median; the mean adds half the residual variance before exponentiating
(exp(x̂ + s²/2)). Output σ is annualised: sqrt(252 × mean daily variance).

Everything here operates on per-name ordered arrays of rv_total (one element per complete session,
in date order). The caller (rv.har's DB code) builds those arrays; these functions never touch a
calendar or a clock, so they are deterministic and unit-tested.
"""

from __future__ import annotations

import numpy as np

from . import MONTH_WINDOW, TRADING_DAYS_PER_YEAR, WEEK_WINDOW


def _trailing_mean(rv: np.ndarray, window: int) -> np.ndarray:
    """Mean of rv[t-window+1 … t] per t; NaN until there are `window` values."""
    n = len(rv)
    out = np.full(n, np.nan)
    if n < window:
        return out
    csum = np.cumsum(rv)
    out[window - 1] = csum[window - 1] / window
    out[window:] = (csum[window:] - csum[:-window]) / window
    return out


def predictors(rv: np.ndarray) -> np.ndarray:
    """
    The (n, 3) matrix [log RV_d, log RV_w, log RV_m] per session t. Rows where the 22-session
    window isn't full yet (t < 21) are NaN. rv_total must be strictly positive on complete days.
    """
    rv = np.asarray(rv, dtype=float)
    log_d = np.log(rv)
    log_w = np.log(_trailing_mean(rv, WEEK_WINDOW))
    log_m = np.log(_trailing_mean(rv, MONTH_WINDOW))
    return np.column_stack([log_d, log_w, log_m])


def forward_log_mean(rv: np.ndarray, h: int) -> np.ndarray:
    """log of the mean of rv[t+1 … t+h] per t; NaN where the next h sessions don't all exist."""
    rv = np.asarray(rv, dtype=float)
    n = len(rv)
    out = np.full(n, np.nan)
    if n <= h:
        return out
    csum = np.cumsum(rv)
    # sum(rv[t+1 … t+h]) = csum[t+h] - csum[t]
    t = np.arange(n - h)
    out[: n - h] = np.log((csum[t + h] - csum[t]) / h)
    return out


def design_row(pred_row: np.ndarray) -> np.ndarray:
    """[1, log RV_d, log RV_w, log RV_m] for one session's predictor triple."""
    return np.concatenate([[1.0], pred_row])


def fit_pooled(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Pooled least squares. `x` is (m, 3) predictor triples, `y` is (m,) log forward means, already
    restricted to finite rows. Returns (beta of length 4 incl. intercept, residual variance s²).
    s² is the mean squared residual on (m - k) degrees of freedom — the variance added back for the
    lognormal mean correction.
    """
    design = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = max(len(y) - design.shape[1], 1)
    s2 = float(resid @ resid) / dof
    return beta, s2


def forecast_sigma(beta: np.ndarray, s2: float, pred_row: np.ndarray) -> float:
    """
    Annualised σ forecast for one session's predictor triple: mean daily variance is
    exp(x̂ + s²/2) with x̂ the fitted log forward-mean variance, then σ = sqrt(252 × that).
    """
    log_var = float(design_row(pred_row) @ beta)
    mean_daily_var = np.exp(log_var + s2 / 2.0)
    return float(np.sqrt(TRADING_DAYS_PER_YEAR * mean_daily_var))
