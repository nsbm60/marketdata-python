"""PR5: invert accounting, capture join, dual lag, residual classes."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from greeks.config import load_config
from greeks.domain import FailureReason, JoinClass, OptionRight, RowStatus
from greeks.forwards.sofr import load_sofr_csv
from greeks.harness.invert_trades import invert_batch, invert_staged_trade
from greeks.harness.join_vendor import (
    build_residual,
    filter_baseline_snapshots,
    join_batch,
    match_capture,
    residual_iv_bps,
    vendor_sane,
)
from greeks.harness.rows import ValidationRow, VendorSnapshot
from greeks.harness.store import ResultsStore
from greeks.pull.staging import StagedTrade
from greeks.solver.black76 import price as black_price
from greeks.solver.time import years_to_expiry
from greeks.forwards.carry import carry
from greeks.forwards.sofr import continuous_rate_from_sofr


def _trade(
    *,
    symbol: str = "NVDA260626C00135000",
    trade_ts: datetime | None = None,
    price: float = 5.0,
    spot: float | None = 135.0,
    session: date = date(2026, 5, 28),
) -> StagedTrade:
    ts = trade_ts or datetime(2026, 5, 28, 15, 0, tzinfo=timezone.utc)
    return StagedTrade(
        symbol=symbol,
        trade_ts=ts,
        session_date=session,
        underlying="NVDA",
        price=price,
        size=1.0,
        exchange=1,
        conditions="",
        sequence_number=1,
        sip_timestamp_ns=int(ts.timestamp() * 1e9),
        spot_at_trade=spot,
        spot_trade_ts=ts - timedelta(seconds=1) if spot else None,
    )


def test_invert_missing_spot() -> None:
    cfg = load_config()
    sofr = load_sofr_csv()
    row = invert_staged_trade(
        _trade(spot=None), cfg=cfg, sofr_series=sofr
    )
    assert row.status is RowStatus.FAILURE
    assert row.reason_code is FailureReason.MISSING_SPOT
    assert row.iv is None


def test_invert_success_and_batch_accounting() -> None:
    cfg = load_config()
    sofr = load_sofr_csv()
    # Build a mid-ish price from Black-76 so invert succeeds
    trade_ts = datetime(2026, 5, 28, 15, 0, tzinfo=timezone.utc)
    session = date(2026, 5, 28)
    expiry = date(2026, 6, 26)
    spot = 135.0
    r = continuous_rate_from_sofr(sofr, session, expiry)
    c = carry(spot, r, trade_ts, expiry, cfg.dividends_for("NVDA"))
    strike = 135.0
    mid = black_price(
        c.forward, strike, c.time_to_expiry, c.discount, 0.35, OptionRight.CALL
    )
    symbol = "NVDA260626C00135000"
    good = _trade(symbol=symbol, trade_ts=trade_ts, price=mid, spot=spot, session=session)
    bad = _trade(symbol=symbol, trade_ts=trade_ts + timedelta(seconds=1), price=mid, spot=None)
    rows, stats = invert_batch([good, bad], cfg=cfg, sofr_series=sofr)
    assert stats.n_input == 2
    assert stats.balanced()
    assert stats.n_success + stats.n_failure == 2
    assert rows[0].status is RowStatus.SUCCESS
    assert rows[0].iv is not None and rows[0].iv > 0
    assert rows[1].reason_code is FailureReason.MISSING_SPOT


def test_match_capture_at_or_after_and_staleness() -> None:
    t0 = datetime(2026, 5, 28, 15, 0, 0, tzinfo=timezone.utc)
    snaps = [
        VendorSnapshot(
            symbol="X",
            underlying="NVDA",
            timestamp=t0 - timedelta(seconds=1),
            quote_timestamp=None,
            bid=1, ask=2, iv=0.3, delta=0.5, gamma=0.01, vega=0.1, theta=-0.05, rho=0.0,
        ),
        VendorSnapshot(
            symbol="X",
            underlying="NVDA",
            timestamp=t0 + timedelta(seconds=30),
            quote_timestamp=t0 + timedelta(seconds=10),
            bid=1, ask=2, iv=0.3, delta=0.5, gamma=0.01, vega=0.1, theta=-0.05, rho=0.0,
        ),
        VendorSnapshot(
            symbol="X",
            underlying="NVDA",
            timestamp=t0 + timedelta(seconds=400),  # > 5 min
            quote_timestamp=None,
            bid=1, ask=2, iv=0.3, delta=0.5, gamma=0.01, vega=0.1, theta=-0.05, rho=0.0,
        ),
    ]
    m = match_capture(t0, snaps, max_staleness_s=300)
    assert m is not None
    assert m.join_lag_capture_ms == 30_000
    assert m.join_lag_quote_ms == 10_000  # distinct from capture lag
    assert m.snapshot.timestamp == t0 + timedelta(seconds=30)

    # Only late snap → miss
    m2 = match_capture(t0, [snaps[2]], max_staleness_s=300)
    assert m2 is None


def test_dual_lags_not_collapsed() -> None:
    t0 = datetime(2026, 5, 28, 15, 0, tzinfo=timezone.utc)
    snap = VendorSnapshot(
        symbol="X",
        underlying="NVDA",
        timestamp=t0 + timedelta(seconds=60),
        quote_timestamp=t0 + timedelta(seconds=5),
        bid=1.0,
        ask=1.1,
        iv=0.25,
        delta=0.4,
        gamma=0.02,
        vega=0.05,
        theta=-0.02,
        rho=-0.001,
    )
    m = match_capture(t0, [snap], max_staleness_s=300)
    assert m is not None
    assert m.join_lag_capture_ms != m.join_lag_quote_ms


def test_vendor_sane_and_wrong_sign() -> None:
    base = dict(
        symbol="X",
        underlying="NVDA",
        timestamp=datetime(2026, 5, 28, 15, 0, tzinfo=timezone.utc),
        quote_timestamp=None,
        bid=1.0,
        ask=1.1,
        iv=0.2,
        delta=0.5,
        gamma=0.01,
        vega=0.1,
        theta=-0.05,
        rho=0.0,
    )
    assert vendor_sane(VendorSnapshot(**base))
    assert not vendor_sane(VendorSnapshot(**{**base, "vega": -0.1}))
    assert not vendor_sane(VendorSnapshot(**{**base, "theta": 0.05}))
    assert not vendor_sane(VendorSnapshot(**{**base, "bid": 0.0}))


def test_join_classes() -> None:
    t0 = datetime(2026, 5, 28, 15, 0, tzinfo=timezone.utc)
    failed = ValidationRow(
        symbol="A",
        trade_ts=t0,
        methodology_version="v1",
        underlying="NVDA",
        expiry=date(2026, 6, 26),
        strike=100.0,
        right=OptionRight.CALL,
        trade_price=1.0,
        spot_at_trade=100.0,
        forward=100.0,
        discount=0.99,
        time_to_expiry=0.08,
        iv=None,
        delta=None,
        gamma=None,
        vega=None,
        theta=None,
        rho=None,
        status=RowStatus.FAILURE,
        reason_code=FailureReason.BELOW_INTRINSIC,
    )
    success = ValidationRow(
        symbol="B",
        trade_ts=t0,
        methodology_version="v1",
        underlying="NVDA",
        expiry=date(2026, 6, 26),
        strike=100.0,
        right=OptionRight.CALL,
        trade_price=5.0,
        spot_at_trade=100.0,
        forward=100.0,
        discount=0.99,
        time_to_expiry=0.08,
        iv=0.30,
        delta=0.5,
        gamma=0.01,
        vega=0.1,
        theta=-0.05,
        rho=-0.001,
        status=RowStatus.SUCCESS,
        reason_code=None,
    )
    r_fail = build_residual(failed, match=None)
    assert r_fail.join_class is JoinClass.OURS_FAILED
    assert r_fail.residual_iv_bps is None

    r_miss = build_residual(success, match=None)
    assert r_miss.join_class is JoinClass.VENDOR_MISSING
    assert r_miss.residual_iv_bps is None  # not disagreement

    snap = VendorSnapshot(
        symbol="B",
        underlying="NVDA",
        timestamp=t0 + timedelta(seconds=20),
        quote_timestamp=t0 + timedelta(seconds=15),
        bid=1.0,
        ask=1.1,
        iv=0.28,
        delta=0.49,
        gamma=0.01,
        vega=0.1,
        theta=-0.04,
        rho=-0.001,
    )
    m = match_capture(t0, [snap])
    r_ok = build_residual(success, match=m)
    assert r_ok.join_class is JoinClass.MATCHED
    assert r_ok.residual_iv_bps == residual_iv_bps(0.30, 0.28)
    assert abs(r_ok.residual_iv_bps - 200.0) < 1e-9  # 200 vol bps
    assert r_ok.join_lag_capture_ms == 20_000
    assert r_ok.join_lag_quote_ms == 15_000


def test_filter_baseline_excludes_bad_day() -> None:
    snaps = [
        VendorSnapshot(
            symbol="X",
            underlying="NVDA",
            timestamp=datetime(2026, 6, 8, 15, 0, tzinfo=timezone.utc),
            quote_timestamp=None,
            bid=1, ask=2, iv=0.2, delta=0.5, gamma=0.01, vega=0.1, theta=-0.01, rho=0,
        ),
        VendorSnapshot(
            symbol="X",
            underlying="NVDA",
            timestamp=datetime(2026, 5, 28, 15, 0, tzinfo=timezone.utc),
            quote_timestamp=None,
            bid=1, ask=2, iv=0.2, delta=0.5, gamma=0.01, vega=0.1, theta=-0.01, rho=0,
        ),
    ]
    kept = filter_baseline_snapshots(
        snaps,
        window_start=date(2026, 5, 27),
        excluded_dates=[date(2026, 6, 8)],
    )
    assert len(kept) == 1
    assert kept[0].timestamp.day == 28


def test_results_store_and_join_batch(tmp_path: Path) -> None:
    cfg = load_config()
    sofr = load_sofr_csv()
    trade_ts = datetime(2026, 5, 28, 15, 0, tzinfo=timezone.utc)
    session = date(2026, 5, 28)
    expiry = date(2026, 6, 26)
    spot = 135.0
    r = continuous_rate_from_sofr(sofr, session, expiry)
    c = carry(spot, r, trade_ts, expiry, cfg.dividends_for("NVDA"))
    mid = black_price(
        c.forward, 135.0, c.time_to_expiry, c.discount, 0.4, OptionRight.CALL
    )
    tr = _trade(
        symbol="NVDA260626C00135000",
        trade_ts=trade_ts,
        price=mid,
        spot=spot,
        session=session,
    )
    rows, stats = invert_batch([tr], cfg=cfg, sofr_series=sofr)
    assert stats.balanced()
    db = tmp_path / "results.db"
    with ResultsStore(db) as store:
        store.upsert_validation(rows)
        counts = store.count_validation(methodology_version="v1")
        assert counts.get("success", 0) + counts.get("failure", 0) == 1

        snap = VendorSnapshot(
            symbol=tr.symbol,
            underlying="NVDA",
            timestamp=trade_ts + timedelta(seconds=12),
            quote_timestamp=trade_ts + timedelta(seconds=8),
            bid=1.0,
            ask=1.2,
            iv=rows[0].iv * 0.99 if rows[0].iv else 0.3,
            delta=0.5,
            gamma=0.01,
            vega=0.1,
            theta=-0.05,
            rho=-0.001,
        )
        residuals = join_batch(rows, {tr.symbol: [snap]}, max_staleness_s=300)
        store.upsert_residuals(residuals)
        jc = store.count_residuals_by_class(methodology_version="v1")
        assert sum(jc.values()) == 1
