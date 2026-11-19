"""Hermetic tests for the Massive analytics validation probe — fake HTTP (httpx.MockTransport) and
pure functions; no live service, no ClickHouse, nothing written."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

import httpx
import pytest

from greeks.validation.massive_analytics import (
    JoinedRow, MassiveRow, OurTrade, atm_line_iv, bucket_delta, bucket_dte, fetch_analytics,
    implied_div, implied_rate, join, our_stats, snapshot_window_utc,
)

_UTC = timezone.utc


def _mrow(ticker: str, *, iv: Optional[float], delta: Optional[float], strike: float,
          expiry: date, day: date, side: str = "call") -> MassiveRow:
    return MassiveRow(
        ticker=ticker, contract_type=side, strike_price=strike, expiration_date=expiry, date=day,
        underlying_price=100.0, option_price=1.0, bid=0.9, ask=1.1, dividend_yield=0.01,
        risk_free_rate=0.05, model="binomial", implied_volatility=iv, delta=delta, gamma=0.01,
        theta=-0.02, vega=0.1, rho=0.05, open_interest=10.0, volume=5.0)


# ── fetch: pagination + errors ───────────────────────────────────────────────────────────────────

def test_fetch_paginates_all_pages() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("cursor") == "2":
            return httpx.Response(200, json={"results": [
                {"ticker": "O:NVDA260417C00110000", "contract_type": "call", "strike_price": 110,
                 "expiration_date": "2026-04-17", "date": "2026-01-05", "underlying_price": 100,
                 "implied_volatility": 0.3}]})
        return httpx.Response(200, json={
            "results": [{"ticker": "O:NVDA260417C00100000", "contract_type": "call", "strike_price": 100,
                         "expiration_date": "2026-04-17", "date": "2026-01-05", "underlying_price": 100,
                         "implied_volatility": 0.25}],
            "next_url": "https://api.massive.com/options/v1/analytics?cursor=2"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows = fetch_analytics("NVDA", date(2026, 1, 5), "KEY", client=client)
    assert [r.strike_price for r in rows] == [100.0, 110.0]


def test_fetch_missing_key_fails_naming_it() -> None:
    with pytest.raises(ValueError, match="MASSIVE_API_KEY"):
        fetch_analytics("NVDA", date(2026, 1, 5), "")


def test_fetch_http_error_fails_with_status_and_body() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404, text="no such underlying")))
    with pytest.raises(RuntimeError, match=r"HTTP 404.*no such underlying"):
        fetch_analytics("NVDA", date(2026, 1, 5), "KEY", client=client, max_attempts=1)


# ── snapshot window: normal and half day ─────────────────────────────────────────────────────────

def test_snapshot_window_normal_day() -> None:
    lo, hi = snapshot_window_utc(date(2026, 11, 25))  # normal Wed, close 16:00 ET → S 15:45 ET (EST)
    assert lo == datetime(2026, 11, 25, 20, 40, tzinfo=_UTC)
    assert hi == datetime(2026, 11, 25, 20, 50, tzinfo=_UTC)


def test_snapshot_window_half_day() -> None:
    lo, hi = snapshot_window_utc(date(2026, 11, 27))  # day after Thanksgiving, close 13:00 → S 12:45 ET
    assert lo == datetime(2026, 11, 27, 17, 40, tzinfo=_UTC)
    assert hi == datetime(2026, 11, 27, 17, 50, tzinfo=_UTC)


# ── min-3 rule + join classification ─────────────────────────────────────────────────────────────

def _trade(iv: float) -> OurTrade:
    return OurTrade(iv=iv, delta=0.5, gamma=0.01, theta=-0.02, vega=0.1,
                    spot=100.0, discount=0.975, tte=0.5, forward=101.0)


def test_join_min_three_rule_and_null_greeks() -> None:
    day, expiry = date(2026, 1, 5), date(2026, 4, 17)
    matched_osi = "NVDA260417C00100000"
    few_osi = "NVDA260417C00110000"
    massive = [
        _mrow(f"O:{matched_osi}", iv=0.26, delta=0.5, strike=100, expiry=expiry, day=day),
        _mrow(f"O:{few_osi}", iv=0.30, delta=0.4, strike=110, expiry=expiry, day=day),
        _mrow("O:NVDA260417C00120000", iv=None, delta=None, strike=120, expiry=expiry, day=day),
    ]
    ours = {matched_osi: [_trade(0.25), _trade(0.26), _trade(0.27)], few_osi: [_trade(0.31), _trade(0.32)]}
    jr = join(massive, ours, day, divs=[])
    assert jr.n_too_few == 1 and jr.n_null_greeks == 1
    assert [r.reason for r in jr.rows] == ["matched", "too_few_trades", "null_greeks"]
    m = jr.matched[0]
    assert m.our_iv == 0.26 and m.massive_iv == 0.26
    assert m.diff("iv") == pytest.approx(0.0, abs=1e-12)


# ── buckets ──────────────────────────────────────────────────────────────────────────────────────

def test_bucket_delta_bands() -> None:
    assert bucket_delta(0.05) == "|d|<0.10"
    assert bucket_delta(0.10) == "0.10-0.25"
    assert bucket_delta(0.25) == "0.25-0.75"
    assert bucket_delta(0.75) == "0.25-0.75"
    assert bucket_delta(0.90) == "0.75-0.90"
    assert bucket_delta(0.95) == "|d|>0.90"


def test_bucket_dte_bands() -> None:
    assert [bucket_dte(d) for d in (0, 2, 3, 7, 8, 30, 31, 90, 91)] == \
        ["0-2", "0-2", "3-7", "3-7", "8-30", "8-30", "31-90", "31-90", ">90"]


# ── implied rate / dividend arithmetic ───────────────────────────────────────────────────────────

def test_implied_rate_and_dividend() -> None:
    import numpy as np
    r, q, t, s = 0.05, 0.02, 0.5, 100.0
    discount = float(np.exp(-r * t))
    assert implied_rate(discount, t) == pytest.approx(r, abs=1e-12)
    forward = s * float(np.exp((r - q) * t))
    assert implied_div(r, forward, s, t) == pytest.approx(q, abs=1e-12)
    assert implied_rate(None, t) is None
    assert implied_div(r, forward, s, 0.0) is None


# ── ATM line fit at k = 0 ────────────────────────────────────────────────────────────────────────

def test_atm_line_fit_reads_intercept_at_k0() -> None:
    import numpy as np
    day, expiry, fwd = date(2026, 1, 5), date(2026, 4, 17), 100.0
    # IV = 0.20 + 0.5 * k, k = ln(K/F); strikes straddle the forward.
    rows = []
    for k_strike in (90.0, 100.0, 110.0):
        k = float(np.log(k_strike / fwd))
        rows.append(_mrow(f"O:NVDA260417C00{int(k_strike)}000", iv=0.20 + 0.5 * k, delta=0.5,
                          strike=k_strike, expiry=expiry, day=day))
    fit = atm_line_iv(rows, expiry, fwd)
    assert fit is not None
    atm_iv, n = fit
    assert atm_iv == pytest.approx(0.20, abs=1e-9) and n == 3


def test_atm_line_needs_straddle_and_three() -> None:
    day, expiry, fwd = date(2026, 1, 5), date(2026, 4, 17), 100.0
    # All strikes above the forward (no straddle) → None.
    rows = [_mrow(f"O:NVDA260417C00{s}000", iv=0.2, delta=0.5, strike=float(s), expiry=expiry, day=day)
            for s in (110, 120, 130)]
    assert atm_line_iv(rows, expiry, fwd) is None
    # Too few → None.
    assert atm_line_iv(rows[:1], expiry, fwd) is None
