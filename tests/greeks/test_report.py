"""PR6: acceptance report gates, buckets, sub-1-DTE, earnings lag slices."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from greeks.domain import FailureReason, JoinClass, OptionRight, RowStatus
from greeks.harness.report import (
    GATE_MEDIAN_ABS_DELTA,
    GATE_MEDIAN_ABS_IV_BPS,
    SUB1_ATM_TARGET_IV_BPS,
    build_report,
    dte_bucket_name,
    evaluate_gate,
    format_report,
    is_gate_row,
    median,
    quote_lag_bucket_name,
)
from greeks.harness.rows import ResidualRow, ValidationRow
from greeks.harness.store import ResultsStore


def _ts(h: int = 15, day: int = 28) -> datetime:
    return datetime(2026, 5, day, h, 0, 0, tzinfo=timezone.utc)


def _residual(
    *,
    symbol: str = "NVDA260626C00135000",
    trade_ts: datetime | None = None,
    mny: float = 1.0,
    dte_years: float = 10.0 / 365.0,
    iv_bps: float = 10.0,
    delta_res: float = 0.001,
    join_class: JoinClass = JoinClass.MATCHED,
    quote_lag_ms: int | None = 15_000,
    underlying: str = "NVDA",
) -> ResidualRow:
    ts = trade_ts or _ts()
    return ResidualRow(
        symbol=symbol,
        trade_ts=ts,
        methodology_version="v1",
        underlying=underlying,
        join_class=join_class,
        snapshot_ts=ts + timedelta(seconds=10),
        join_lag_capture_ms=10_000,
        quote_ts=ts + timedelta(milliseconds=quote_lag_ms or 0),
        join_lag_quote_ms=quote_lag_ms,
        our_iv=0.30,
        our_delta=0.5,
        our_gamma=0.01,
        our_vega=0.1,
        our_theta=-0.05,
        our_rho=-0.001,
        vendor_iv=0.30 - iv_bps / 10_000.0,
        vendor_delta=0.5 - delta_res,
        vendor_gamma=0.01,
        vendor_vega=0.1,
        vendor_theta=-0.05,
        vendor_rho=-0.001,
        residual_iv_bps=iv_bps,
        residual_delta=delta_res,
        residual_gamma=0.0,
        residual_vega=0.0,
        residual_theta=0.0,
        residual_rho=0.0,
        moneyness=mny,
        dte_years=dte_years,
    )


def _val(
    *,
    symbol: str = "NVDA260626C00135000",
    trade_ts: datetime | None = None,
    expiry: date = date(2026, 6, 26),
    status: RowStatus = RowStatus.SUCCESS,
    reason: FailureReason | None = None,
) -> ValidationRow:
    ts = trade_ts or _ts()
    return ValidationRow(
        symbol=symbol,
        trade_ts=ts,
        methodology_version="v1",
        underlying="NVDA",
        expiry=expiry,
        strike=135.0,
        right=OptionRight.CALL,
        trade_price=5.0,
        spot_at_trade=135.0,
        forward=135.0,
        discount=0.99,
        time_to_expiry=0.08,
        iv=0.3 if status is RowStatus.SUCCESS else None,
        delta=0.5 if status is RowStatus.SUCCESS else None,
        gamma=0.01 if status is RowStatus.SUCCESS else None,
        vega=0.1 if status is RowStatus.SUCCESS else None,
        theta=-0.05 if status is RowStatus.SUCCESS else None,
        rho=-0.001 if status is RowStatus.SUCCESS else None,
        status=status,
        reason_code=reason,
    )


def test_median() -> None:
    assert median([1.0, 3.0, 2.0]) == 2.0
    assert median([1.0, 2.0]) == 1.5
    assert median([]) is None


def test_dte_buckets() -> None:
    assert dte_bucket_name(1.0 / 365.0 / 24.0) == "<2h"  # 1 hour
    assert dte_bucket_name(4.0 / 365.0 / 24.0) == "2h-6.5h"
    assert dte_bucket_name(0.5 / 365.0) == "6.5h-1d"
    assert dte_bucket_name(10.0 / 365.0) == "7d-30d"


def test_gate_pass() -> None:
    rows = [
        _residual(iv_bps=10.0, delta_res=0.002, dte_years=10 / 365, mny=1.0),
        _residual(iv_bps=12.0, delta_res=0.003, dte_years=15 / 365, mny=1.0),
        _residual(iv_bps=8.0, delta_res=0.001, dte_years=20 / 365, mny=0.98),
    ]
    g = evaluate_gate(rows)
    assert g.n == 3
    assert g.gated is True
    assert g.pass_gate is True
    assert g.median_abs_iv_bps is not None
    assert g.median_abs_iv_bps <= GATE_MEDIAN_ABS_IV_BPS
    assert g.median_abs_delta is not None
    assert g.median_abs_delta <= GATE_MEDIAN_ABS_DELTA


def test_gate_fail_iv() -> None:
    rows = [
        _residual(iv_bps=50.0, delta_res=0.001, dte_years=10 / 365),
        _residual(iv_bps=60.0, delta_res=0.001, dte_years=12 / 365),
    ]
    g = evaluate_gate(rows)
    assert g.pass_gate is False


def test_vendor_missing_not_in_gate() -> None:
    rows = [
        _residual(iv_bps=10.0, join_class=JoinClass.VENDOR_MISSING),
        _residual(iv_bps=10.0, dte_years=10 / 365),
    ]
    # vendor_missing has residuals set but join_class excludes it
    rows[0] = ResidualRow(
        **{
            **rows[0].__dict__,
            "join_class": JoinClass.VENDOR_MISSING,
            "residual_iv_bps": None,
            "residual_delta": None,
        }
    )
    g = evaluate_gate(rows)
    assert g.n == 1


def test_is_gate_row_bounds() -> None:
    assert is_gate_row(_residual(mny=1.0, dte_years=10 / 365))
    assert not is_gate_row(_residual(mny=1.0, dte_years=1 / 365))  # too short
    assert not is_gate_row(_residual(mny=1.2, dte_years=10 / 365))  # wing


def test_full_report_pass() -> None:
    val = [_val(), _val(symbol="NVDA260626P00135000", trade_ts=_ts(16))]
    res = [
        _residual(dte_years=10 / 365, iv_bps=5.0),
        _residual(
            symbol="NVDA260626P00135000",
            trade_ts=_ts(16),
            dte_years=12 / 365,
            iv_bps=7.0,
        ),
    ]
    rep = build_report(
        methodology_version="v1", validation=val, residuals=res
    )
    assert rep.validation_accounting_ok
    assert rep.gate.pass_gate is True
    assert rep.overall_pass is True
    text = format_report(rep)
    assert "PASS" in text
    assert "GATE_NTM" in text


def test_sub1dte_accounting_and_target() -> None:
    # Expiry day = 2026-05-28
    expiry = date(2026, 5, 28)
    ts_am = datetime(2026, 5, 28, 14, 0, tzinfo=timezone.utc)  # ~2h+ if expiry 16:00 ET
    # dte ~ 6 hours in years
    dte_6h = 6.0 / 24.0 / 365.0
    dte_1h = 1.0 / 24.0 / 365.0
    val = [
        _val(trade_ts=ts_am, expiry=expiry, status=RowStatus.SUCCESS),
        _val(
            symbol="NVDA260528C00135000",
            trade_ts=ts_am + timedelta(minutes=1),
            expiry=expiry,
            status=RowStatus.FAILURE,
            reason=FailureReason.T_BELOW_FLOOR,
        ),
    ]
    res = [
        _residual(
            symbol=val[0].symbol,
            trade_ts=ts_am,
            mny=1.0,
            dte_years=dte_6h,
            iv_bps=40.0,  # meets 100 bps target
        ),
        _residual(
            symbol="WING",
            trade_ts=ts_am,
            mny=1.20,
            dte_years=dte_6h,
            iv_bps=200.0,
        ),
        _residual(
            symbol="ATM2H",
            trade_ts=ts_am,
            mny=1.0,
            dte_years=dte_1h,
            iv_bps=150.0,
        ),
    ]
    # Align symbols for expiry map
    res = [
        _residual(
            symbol=val[0].symbol,
            trade_ts=ts_am,
            mny=1.0,
            dte_years=dte_6h,
            iv_bps=40.0,
        ),
        _residual(
            symbol="NVDA260528C00150000",
            trade_ts=ts_am,
            mny=1.20,
            dte_years=dte_6h,
            iv_bps=200.0,
        ),
        _residual(
            symbol="NVDA260528C00135001",
            trade_ts=ts_am,
            mny=1.0,
            dte_years=dte_1h,
            iv_bps=150.0,
        ),
    ]
    # validation needs rows for wing/atm symbols for expiry-day map — use fallback dte
    rep = build_report(methodology_version="v1", validation=val, residuals=res)
    assert rep.sub1dte.accounting_ok
    assert rep.sub1dte.n_validation_expiry_day == 2
    assert rep.sub1dte.n_success == 1
    assert rep.sub1dte.n_failure == 1
    assert rep.sub1dte.atm_gt_2h.n >= 1
    assert rep.sub1dte.atm_gt_2h.median_abs_iv_bps is not None
    assert rep.sub1dte.atm_gt_2h.median_abs_iv_bps <= SUB1_ATM_TARGET_IV_BPS


def test_earnings_quote_lag_buckets() -> None:
    ed = date(2026, 6, 1)
    # In window T-1
    ts = datetime(2026, 5, 31, 15, 0, tzinfo=timezone.utc)
    res = [
        _residual(trade_ts=ts, quote_lag_ms=3_000, iv_bps=25.0, dte_years=20 / 365),
        _residual(
            symbol="B",
            trade_ts=ts + timedelta(seconds=1),
            quote_lag_ms=90_000,
            iv_bps=40.0,
            dte_years=20 / 365,
        ),
        # Outside window
        _residual(
            symbol="C",
            trade_ts=datetime(2026, 5, 20, 15, 0, tzinfo=timezone.utc),
            quote_lag_ms=3_000,
            iv_bps=5.0,
            dte_years=20 / 365,
        ),
    ]
    val = [_val(symbol=r.symbol, trade_ts=r.trade_ts) for r in res]
    rep = build_report(
        methodology_version="v1",
        validation=val,
        residuals=res,
        earnings_events=[("NVDA", ed)],
    )
    assert rep.earnings is not None
    assert rep.earnings.n_matched_in_window == 2
    assert rep.earnings.baseline_matched.n == 1
    names = {b.name for b in rep.earnings.by_quote_lag}
    assert "quote_lag_<5s" in names
    assert "quote_lag_1-5m" in names


def test_quote_lag_bucket_names() -> None:
    assert quote_lag_bucket_name(None) == "quote_lag_missing"
    assert quote_lag_bucket_name(1000) == "quote_lag_<5s"
    assert quote_lag_bucket_name(200_000) == "quote_lag_1-5m"


def test_store_roundtrip_report(tmp_path: Path) -> None:
    db = tmp_path / "r.db"
    val = [_val()]
    res = [_residual(dte_years=10 / 365, iv_bps=5.0)]
    with ResultsStore(db) as store:
        store.upsert_validation(val)
        store.upsert_residuals(res)
        loaded_v = store.load_validation(methodology_version="v1")
        loaded_r = store.load_residuals(methodology_version="v1")
    rep = build_report(
        methodology_version="v1", validation=loaded_v, residuals=loaded_r
    )
    assert rep.gate.n == 1
    assert rep.overall_pass is True
