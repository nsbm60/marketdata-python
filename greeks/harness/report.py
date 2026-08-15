"""Acceptance reporting for greeks validation (PR6).

**Hard gate (Tier-3 NTM short-dated):** matched residuals with moneyness in
``[0.95, 1.05]`` and DTE in ``[3, 30]`` calendar days:

- median ``|residual_iv_bps|`` ≤ 20
- median ``|residual_delta|`` ≤ 0.005

Other buckets, sub-1-DTE ATM target (100 bps), and earnings slices are
**reported, not gated**.

``vendor_missing`` is never counted as disagreement (excluded from residual
medians; only ``matched`` rows contribute).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Mapping, Optional, Sequence

from greeks.domain import JoinClass, RowStatus
from greeks.harness.rows import ResidualRow, ValidationRow

# --- thresholds (brief / plan) ---
GATE_MNY_LO = 0.95
GATE_MNY_HI = 1.05
GATE_DTE_LO_DAYS = 3.0
GATE_DTE_HI_DAYS = 30.0
GATE_MEDIAN_ABS_IV_BPS = 20.0
GATE_MEDIAN_ABS_DELTA = 0.005

SUB1_ATM_MNY_LO = 0.97
SUB1_ATM_MNY_HI = 1.03
SUB1_ATM_TARGET_IV_BPS = 100.0  # reported target, not hard gate
SUB1_MIN_HOURS_TO_EXPIRY = 2.0

EARNINGS_WINDOW_BEFORE = 3  # T-3
EARNINGS_WINDOW_AFTER = 2  # T+2


@dataclass(frozen=True)
class BucketStats:
    name: str
    n: int
    median_abs_iv_bps: Optional[float]
    median_abs_delta: Optional[float]
    gated: bool = False
    pass_gate: Optional[bool] = None


@dataclass(frozen=True)
class ReasonCount:
    reason: str
    n: int


@dataclass(frozen=True)
class Sub1DteReport:
    """Expiry-day accounting + ATM intraday residual (report-only target)."""

    n_validation_expiry_day: int
    n_success: int
    n_failure: int
    accounting_ok: bool  # success + failure == total
    by_reason: tuple[ReasonCount, ...]
    # Matched residuals on expiry day
    atm_gt_2h: BucketStats  # 0.97–1.03, T>2h — target 100 bps
    atm_final_2h: BucketStats  # 0.97–1.03, T≤2h — report only
    wings: BucketStats  # outside ATM moneyness — report only


@dataclass(frozen=True)
class LagBucketStats:
    name: str
    n: int
    median_abs_iv_bps: Optional[float]


@dataclass(frozen=True)
class EarningsSliceReport:
    """T-3..T+2 around earnings; residuals bucketed by quote-clock lag."""

    n_events: int
    n_matched_in_window: int
    by_quote_lag: tuple[LagBucketStats, ...]
    baseline_matched: BucketStats  # outside earnings windows (same underlyings)


@dataclass(frozen=True)
class AcceptanceReport:
    methodology_version: str
    n_validation: int
    n_validation_success: int
    n_validation_failure: int
    validation_accounting_ok: bool
    join_class_counts: Mapping[str, int]
    gate: BucketStats
    buckets: tuple[BucketStats, ...]
    sub1dte: Sub1DteReport
    earnings: Optional[EarningsSliceReport]
    overall_pass: bool  # hard gate only


def median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def dte_days(dte_years: float) -> float:
    return dte_years * 365.0


def dte_hours(dte_years: float) -> float:
    return dte_years * 365.0 * 24.0


def dte_bucket_name(dte_years: float) -> str:
    """Intraday + multi-day DTE buckets (plan PR6)."""
    h = dte_hours(dte_years)
    d = dte_days(dte_years)
    if h < 2.0:
        return "<2h"
    if h < 6.5:
        return "2h-6.5h"
    if d < 1.0:
        return "6.5h-1d"
    if d < 3.0:
        return "1d-3d"
    if d < 7.0:
        return "3d-7d"
    if d < 30.0:
        return "7d-30d"
    if d <= 90.0:
        return "30d-90d"
    return ">90d"


def moneyness_bucket_name(mny: float) -> str:
    if mny < 0.85:
        return "mny<0.85"
    if mny < 0.95:
        return "0.85-0.95"
    if mny <= 1.05:
        return "0.95-1.05"
    if mny <= 1.15:
        return "1.05-1.15"
    return "mny>1.15"


def is_gate_row(row: ResidualRow) -> bool:
    if row.join_class is not JoinClass.MATCHED:
        return False
    if row.moneyness is None or row.dte_years is None:
        return False
    if row.residual_iv_bps is None or row.residual_delta is None:
        return False
    d = dte_days(row.dte_years)
    return (
        GATE_MNY_LO <= row.moneyness <= GATE_MNY_HI
        and GATE_DTE_LO_DAYS <= d <= GATE_DTE_HI_DAYS
    )


def matched_only(rows: Sequence[ResidualRow]) -> list[ResidualRow]:
    return [
        r
        for r in rows
        if r.join_class is JoinClass.MATCHED
        and r.residual_iv_bps is not None
        and r.residual_delta is not None
        and r.moneyness is not None
        and r.dte_years is not None
    ]


def stats_from_rows(
    name: str,
    rows: Sequence[ResidualRow],
    *,
    gated: bool = False,
    iv_limit: Optional[float] = None,
    delta_limit: Optional[float] = None,
) -> BucketStats:
    ivs = [abs(r.residual_iv_bps) for r in rows if r.residual_iv_bps is not None]
    deltas = [abs(r.residual_delta) for r in rows if r.residual_delta is not None]
    med_iv = median(ivs)
    med_d = median(deltas)
    pass_gate: Optional[bool] = None
    if gated and med_iv is not None and med_d is not None:
        assert iv_limit is not None and delta_limit is not None
        pass_gate = med_iv <= iv_limit and med_d <= delta_limit
    elif gated:
        pass_gate = False  # empty gate bucket fails closed
    return BucketStats(
        name=name,
        n=len(rows),
        median_abs_iv_bps=med_iv,
        median_abs_delta=med_d,
        gated=gated,
        pass_gate=pass_gate,
    )


def bucket_matched(rows: Sequence[ResidualRow]) -> list[BucketStats]:
    """Cross moneyness × DTE buckets (matched only)."""
    matched = matched_only(rows)
    groups: dict[str, list[ResidualRow]] = {}
    for r in matched:
        assert r.moneyness is not None and r.dte_years is not None
        key = f"{moneyness_bucket_name(r.moneyness)}|{dte_bucket_name(r.dte_years)}"
        groups.setdefault(key, []).append(r)
    out = [stats_from_rows(k, v) for k, v in sorted(groups.items())]
    return out


def evaluate_gate(rows: Sequence[ResidualRow]) -> BucketStats:
    gate_rows = [r for r in matched_only(rows) if is_gate_row(r)]
    return stats_from_rows(
        "GATE_NTM_3-30DTE",
        gate_rows,
        gated=True,
        iv_limit=GATE_MEDIAN_ABS_IV_BPS,
        delta_limit=GATE_MEDIAN_ABS_DELTA,
    )


def validation_accounting(
    rows: Sequence[ValidationRow],
) -> tuple[int, int, int, bool, tuple[ReasonCount, ...]]:
    n = len(rows)
    n_ok = sum(1 for r in rows if r.status is RowStatus.SUCCESS)
    n_fail = sum(1 for r in rows if r.status is RowStatus.FAILURE)
    by: dict[str, int] = {}
    for r in rows:
        if r.status is RowStatus.FAILURE:
            key = r.reason_code.value if r.reason_code else "unknown"
            by[key] = by.get(key, 0) + 1
    reasons = tuple(ReasonCount(k, by[k]) for k in sorted(by))
    return n, n_ok, n_fail, n == n_ok + n_fail, reasons


def build_sub1dte_report(
    validation: Sequence[ValidationRow],
    residuals: Sequence[ResidualRow],
) -> Sub1DteReport:
    """Expiry-day rows: trade calendar date == expiry date."""
    exp_val = [
        r
        for r in validation
        if r.trade_ts.astimezone(timezone.utc).date() == r.expiry
    ]
    n, n_ok, n_fail, ok, reasons = validation_accounting(exp_val)

    val_expiry = {(r.symbol, r.trade_ts): r.expiry for r in validation}
    exp_matched: list[ResidualRow] = []
    for r in matched_only(residuals):
        exp = val_expiry.get((r.symbol, r.trade_ts))
        if exp is None:
            # fallback: sub-1 calendar day of life
            if r.dte_years is not None and dte_days(r.dte_years) < 1.0:
                exp_matched.append(r)
            continue
        if r.trade_ts.astimezone(timezone.utc).date() == exp:
            exp_matched.append(r)

    atm_gt_2h: list[ResidualRow] = []
    atm_final_2h: list[ResidualRow] = []
    wings: list[ResidualRow] = []
    for r in exp_matched:
        assert r.moneyness is not None and r.dte_years is not None
        h = dte_hours(r.dte_years)
        atm = SUB1_ATM_MNY_LO <= r.moneyness <= SUB1_ATM_MNY_HI
        if not atm:
            wings.append(r)
        elif h > SUB1_MIN_HOURS_TO_EXPIRY:
            atm_gt_2h.append(r)
        else:
            atm_final_2h.append(r)

    return Sub1DteReport(
        n_validation_expiry_day=n,
        n_success=n_ok,
        n_failure=n_fail,
        accounting_ok=ok,
        by_reason=reasons,
        atm_gt_2h=stats_from_rows(
            "sub1dte_ATM_gt2h (target≤100bps, report-only)",
            atm_gt_2h,
        ),
        atm_final_2h=stats_from_rows("sub1dte_ATM_final2h (report-only)", atm_final_2h),
        wings=stats_from_rows("sub1dte_wings (report-only)", wings),
    )


def quote_lag_bucket_name(lag_ms: Optional[int]) -> str:
    if lag_ms is None:
        return "quote_lag_missing"
    s = abs(lag_ms) / 1000.0
    if s < 5:
        return "quote_lag_<5s"
    if s < 30:
        return "quote_lag_5-30s"
    if s < 60:
        return "quote_lag_30-60s"
    if s < 300:
        return "quote_lag_1-5m"
    return "quote_lag_>5m"


def build_earnings_report(
    residuals: Sequence[ResidualRow],
    earnings_events: Sequence[tuple[str, date]],
    *,
    before: int = EARNINGS_WINDOW_BEFORE,
    after: int = EARNINGS_WINDOW_AFTER,
) -> EarningsSliceReport:
    """Residuals on underlyings within [T-before, T+after] of an earnings date.

    Bucketed by **quote-clock** join lag (PR5), not capture lag.
    """
    windows: list[tuple[str, date, date]] = []
    for und, ed in earnings_events:
        lo = ed - timedelta(days=before)
        hi = ed + timedelta(days=after)
        windows.append((und.upper(), lo, hi))

    def in_earnings(r: ResidualRow) -> bool:
        d = r.trade_ts.astimezone(timezone.utc).date()
        u = r.underlying.upper()
        for und, lo, hi in windows:
            if u == und and lo <= d <= hi:
                return True
        return False

    matched = matched_only(residuals)
    unds = {u for u, _, _ in windows}
    in_win = [r for r in matched if in_earnings(r)]
    baseline = [
        r
        for r in matched
        if r.underlying.upper() in unds and not in_earnings(r)
    ]

    lag_groups: dict[str, list[ResidualRow]] = {}
    for r in in_win:
        key = quote_lag_bucket_name(r.join_lag_quote_ms)
        lag_groups.setdefault(key, []).append(r)

    by_lag = tuple(
        LagBucketStats(
            name=k,
            n=len(v),
            median_abs_iv_bps=median(
                [abs(x.residual_iv_bps) for x in v if x.residual_iv_bps is not None]
            ),
        )
        for k, v in sorted(lag_groups.items())
    )

    return EarningsSliceReport(
        n_events=len(earnings_events),
        n_matched_in_window=len(in_win),
        by_quote_lag=by_lag,
        baseline_matched=stats_from_rows("earnings_baseline_matched", baseline),
    )


def build_report(
    *,
    methodology_version: str,
    validation: Sequence[ValidationRow],
    residuals: Sequence[ResidualRow],
    earnings_events: Sequence[tuple[str, date]] = (),
) -> AcceptanceReport:
    n, n_ok, n_fail, val_ok, _ = validation_accounting(validation)
    jc: dict[str, int] = {}
    for r in residuals:
        jc[r.join_class.value] = jc.get(r.join_class.value, 0) + 1

    gate = evaluate_gate(residuals)
    buckets = tuple(bucket_matched(residuals))
    sub1 = build_sub1dte_report(validation, residuals)
    earn: Optional[EarningsSliceReport] = None
    if earnings_events:
        earn = build_earnings_report(residuals, earnings_events)

    # Hard gate is NTM short-dated only. Validation accounting (AC5) is required.
    # Sub-1-DTE ATM 100 bps target is report-only and does not fail overall_pass.
    if not val_ok:
        overall = False
    elif gate.n == 0:
        overall = False  # no gate data → cannot claim pass
    else:
        overall = bool(gate.pass_gate)

    return AcceptanceReport(
        methodology_version=methodology_version,
        n_validation=n,
        n_validation_success=n_ok,
        n_validation_failure=n_fail,
        validation_accounting_ok=val_ok,
        join_class_counts=jc,
        gate=gate,
        buckets=buckets,
        sub1dte=sub1,
        earnings=earn,
        overall_pass=overall,
    )


def format_report(report: AcceptanceReport) -> str:
    lines: list[str] = []
    lines.append(f"=== Greeks validation report (methodology={report.methodology_version}) ===")
    lines.append(
        f"validation: n={report.n_validation} success={report.n_validation_success} "
        f"failure={report.n_validation_failure} accounting_ok={report.validation_accounting_ok}"
    )
    lines.append(f"join_class: {dict(report.join_class_counts)}")
    lines.append("")
    lines.append("--- HARD GATE (NTM 0.95–1.05, 3–30 DTE) ---")
    lines.append(_fmt_bucket(report.gate))
    gate_status = (
        "PASS"
        if report.gate.pass_gate
        else ("FAIL" if report.gate.n > 0 else "NO_DATA")
    )
    lines.append(f"gate_result: {gate_status}")
    lines.append("")
    lines.append("--- All matched buckets (moneyness|DTE) — report only ---")
    if not report.buckets:
        lines.append("  (no matched residual rows)")
    for b in report.buckets:
        lines.append(_fmt_bucket(b))
    lines.append("")
    lines.append("--- Sub-1-DTE (expiry day) ---")
    s = report.sub1dte
    lines.append(
        f"  expiry_day validation: n={s.n_validation_expiry_day} "
        f"success={s.n_success} failure={s.n_failure} accounting_ok={s.accounting_ok}"
    )
    if s.by_reason:
        lines.append(
            "  failure reasons: "
            + ", ".join(f"{r.reason}={r.n}" for r in s.by_reason)
        )
    lines.append(f"  {_fmt_bucket(s.atm_gt_2h)}")
    if s.atm_gt_2h.median_abs_iv_bps is not None:
        hit = s.atm_gt_2h.median_abs_iv_bps <= SUB1_ATM_TARGET_IV_BPS
        lines.append(
            f"    target |IV|≤{SUB1_ATM_TARGET_IV_BPS:.0f} bps: "
            f"{'MEETS_TARGET' if hit else 'ABOVE_TARGET'} (not hard-gated)"
        )
    lines.append(f"  {_fmt_bucket(s.atm_final_2h)}")
    lines.append(f"  {_fmt_bucket(s.wings)}")
    lines.append("")
    if report.earnings is not None:
        e = report.earnings
        lines.append("--- Earnings slice T-3..T+2 (report only; quote-clock lag) ---")
        lines.append(
            f"  events={e.n_events} matched_in_window={e.n_matched_in_window}"
        )
        lines.append(f"  {_fmt_bucket(e.baseline_matched)}")
        for lb in e.by_quote_lag:
            iv = (
                f"{lb.median_abs_iv_bps:.2f}"
                if lb.median_abs_iv_bps is not None
                else "n/a"
            )
            lines.append(f"  {lb.name}: n={lb.n} median_|iv_bps|={iv}")
        lines.append("")
    lines.append(
        f"OVERALL (hard gate + validation accounting): "
        f"{'PASS' if report.overall_pass else 'FAIL'}"
    )
    return "\n".join(lines)


def _fmt_bucket(b: BucketStats) -> str:
    iv = f"{b.median_abs_iv_bps:.2f}" if b.median_abs_iv_bps is not None else "n/a"
    d = f"{b.median_abs_delta:.6f}" if b.median_abs_delta is not None else "n/a"
    extra = ""
    if b.gated:
        extra = f" pass={b.pass_gate}"
    return f"  {b.name}: n={b.n} median_|iv_bps|={iv} median_|delta|={d}{extra}"


# ClickHouse SQL template (parameterize methodology_version in client).
CH_RESIDUAL_BUCKET_SQL = """
-- Matched residuals only; IV residual already in vol bps.
-- Run against trading.greeks_residuals after PR5 load.
SELECT
    multiIf(
        dte_years * 365 * 24 < 2, '<2h',
        dte_years * 365 * 24 < 6.5, '2h-6.5h',
        dte_years * 365 < 1, '6.5h-1d',
        dte_years * 365 < 3, '1d-3d',
        dte_years * 365 < 7, '3d-7d',
        dte_years * 365 < 30, '7d-30d',
        dte_years * 365 <= 90, '30d-90d',
        '>90d'
    ) AS dte_bucket,
    multiIf(
        moneyness < 0.85, 'mny<0.85',
        moneyness < 0.95, '0.85-0.95',
        moneyness <= 1.05, '0.95-1.05',
        moneyness <= 1.15, '1.05-1.15',
        'mny>1.15'
    ) AS mny_bucket,
    count() AS n,
    median(abs(residual_iv_bps)) AS median_abs_iv_bps,
    median(abs(residual_delta)) AS median_abs_delta
FROM trading.greeks_residuals
WHERE methodology_version = {methodology_version:String}
  AND join_class = 'matched'
  AND residual_iv_bps IS NOT NULL
  AND residual_delta IS NOT NULL
GROUP BY dte_bucket, mny_bucket
ORDER BY mny_bucket, dte_bucket
"""

CH_GATE_SQL = """
SELECT
    count() AS n,
    median(abs(residual_iv_bps)) AS median_abs_iv_bps,
    median(abs(residual_delta)) AS median_abs_delta,
    median(abs(residual_iv_bps)) <= 20 AND median(abs(residual_delta)) <= 0.005 AS gate_pass
FROM trading.greeks_residuals
WHERE methodology_version = {methodology_version:String}
  AND join_class = 'matched'
  AND moneyness BETWEEN 0.95 AND 1.05
  AND dte_years * 365 BETWEEN 3 AND 30
  AND residual_iv_bps IS NOT NULL
  AND residual_delta IS NOT NULL
"""
