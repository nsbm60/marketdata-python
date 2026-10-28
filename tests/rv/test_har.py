"""Offline tests for the HAR numeric core (rv/har.py) — no ClickHouse."""

import numpy as np

from rv import har


def test_trailing_mean_matches_definition():
    rv = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = har._trailing_mean(rv, 3)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == (1 + 2 + 3) / 3
    assert out[3] == (2 + 3 + 4) / 3
    assert out[4] == (3 + 4 + 5) / 3


def test_predictors_log_of_level_means():
    rv = np.arange(1.0, 30.0)  # strictly positive
    p = har.predictors(rv)
    # First 21 rows have no full 22-window -> month column NaN.
    assert np.all(np.isnan(p[:21, 2]))
    assert np.isfinite(p[21:, :]).all()
    # RV_d is just log(rv).
    assert np.allclose(p[:, 0], np.log(rv))
    # RV_w at t=4 is log(mean(rv[0:5])).
    assert np.isclose(p[4, 1], np.log(rv[0:5].mean()))


def test_forward_log_mean_is_next_h_average():
    rv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    f = har.forward_log_mean(rv, 2)
    # t=0 -> mean(rv[1:3]) = mean(2,3)
    assert np.isclose(f[0], np.log((2 + 3) / 2))
    # t=3 -> mean(rv[4:6]) = mean(5,6)
    assert np.isclose(f[3], np.log((5 + 6) / 2))
    # last two have no full horizon
    assert np.isnan(f[4]) and np.isnan(f[5])


def test_fit_recovers_known_linear_relation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5000, 3))
    true_beta = np.array([0.3, 0.5, 0.2, 0.1])
    noise = rng.normal(scale=0.1, size=5000)
    y = true_beta[0] + x @ true_beta[1:] + noise
    beta, s2 = har.fit_pooled(x, y)
    assert np.allclose(beta, true_beta, atol=0.02)
    assert np.isclose(s2, 0.01, atol=0.003)  # noise variance ~0.1^2


def test_forecast_sigma_matches_closed_form():
    beta = np.array([-9.0, 0.5, 0.3, 0.2])
    s2 = 0.4
    pred = np.array([-9.0, -9.0, -9.0])  # log daily variance around exp(-9) ~ 1.2e-4
    log_var = beta[0] + beta[1:] @ pred
    expected = np.sqrt(252 * np.exp(log_var + s2 / 2))
    assert np.isclose(har.forecast_sigma(beta, s2, pred), expected)


def test_har_beats_trailing_mean_on_har_generated_data():
    # Generate log-RV with HAR-like persistence; HAR should forecast the 5-day forward mean better
    # (lower QLIKE) than the naive trailing-22 level.
    rng = np.random.default_rng(7)
    n = 3000
    log_rv = np.zeros(n)
    log_rv[0] = -9.0
    for t in range(1, n):
        w = log_rv[max(0, t - 5) : t].mean()
        m = log_rv[max(0, t - 22) : t].mean()
        log_rv[t] = 0.2 * log_rv[t - 1] + 0.5 * w + 0.3 * m - 9.0 * 0.0 + rng.normal(scale=0.3)
    rv = np.exp(log_rv)
    h = 5
    pred = har.predictors(rv)
    tgt = har.forward_log_mean(rv, h)
    ok = np.isfinite(tgt) & np.isfinite(pred).all(axis=1)
    half = n // 2
    tr = ok.copy()
    tr[half:] = False
    beta, s2 = har.fit_pooled(pred[tr], tgt[tr])

    # Evaluate on the held-out second half.
    ev = ok.copy()
    ev[:half] = False
    realized = np.exp(tgt[ev])  # realized average daily variance over the horizon
    har_var = np.array([np.exp(har.design_row(p) @ beta + s2 / 2) for p in pred[ev]])
    base_var = np.exp(pred[ev][:, 2])  # trailing-22 level = exp(log RV_m)

    def qlike(real, fvar):
        r = real / fvar
        return float(np.mean(r - np.log(r) - 1))

    assert qlike(realized, har_var) < qlike(realized, base_var)
