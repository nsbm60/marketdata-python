"""Join our validation rows to vendor ``option_snapshot``.

**Match membership:** nearest capture clock ``timestamp`` **at-or-after** trade
time, within ``max_staleness_s`` (default 300s / 5 min).

**Lags (two columns — never collapsed):**
- ``join_lag_capture_ms`` = capture_ts - trade_ts (ms)
- ``join_lag_quote_ms`` = quote_ts - trade_ts (ms) when quote present
  (earnings-slice lag metric in PR6; match still uses capture)

**Baseline:** caller filters snapshots to date ≥ window start and not in
``excluded_dates`` before calling (see :func:`filter_baseline_snapshots`).

**Vendor sanity (for MATCHED):** two-sided quote; ``|delta| ≤ 1``; vega ≥ 0;
theta ≤ 0. Failures of sanity → ``vendor_missing`` (not disagreement).

**Three-way class:** ``matched`` | ``vendor_missing`` | ``ours_failed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Mapping, Optional, Sequence

from greeks.domain import JoinClass, RowStatus
from greeks.harness.rows import ResidualRow, ValidationRow, VendorSnapshot


@dataclass(frozen=True)
class CaptureMatch:
    snapshot: VendorSnapshot
    join_lag_capture_ms: int
    join_lag_quote_ms: Optional[int]


def _aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return ts.astimezone(timezone.utc)


def lag_ms(later: datetime, earlier: datetime) -> int:
    return int((_aware(later) - _aware(earlier)).total_seconds() * 1000.0)


def vendor_sane(snap: VendorSnapshot) -> bool:
    """Two-sided quote, |delta|≤1, non-negative vega, non-positive theta."""
    if snap.bid is None or snap.ask is None:
        return False
    if snap.bid <= 0 or snap.ask <= 0:
        return False
    if snap.ask < snap.bid:
        return False
    if snap.iv is None or snap.iv <= 0:
        return False
    if snap.delta is None or abs(snap.delta) > 1.0:
        return False
    if snap.vega is None or snap.vega < 0:
        return False
    if snap.theta is None or snap.theta > 0:
        return False
    return True


def match_capture(
    trade_ts: datetime,
    snapshots: Sequence[VendorSnapshot],
    *,
    max_staleness_s: int = 300,
) -> Optional[CaptureMatch]:
    """Nearest capture timestamp at-or-after trade, within max staleness.

    ``snapshots`` must be for the same symbol; order does not matter.
    """
    t0 = _aware(trade_ts)
    max_ms = max_staleness_s * 1000
    best: Optional[VendorSnapshot] = None
    best_lag = max_ms + 1
    for snap in snapshots:
        cap = _aware(snap.timestamp)
        if cap < t0:
            continue
        lag = lag_ms(cap, t0)
        if lag > max_ms:
            continue
        if best is None or lag < best_lag:
            best = snap
            best_lag = lag
    if best is None:
        return None
    q_lag: Optional[int] = None
    if best.quote_timestamp is not None:
        q_lag = lag_ms(best.quote_timestamp, t0)
    return CaptureMatch(
        snapshot=best,
        join_lag_capture_ms=best_lag,
        join_lag_quote_ms=q_lag,
    )


def residual_iv_bps(our_iv: float, vendor_iv: float) -> float:
    """IV residual in vol bps: 10_000 * (our - vendor)."""
    return 10_000.0 * (our_iv - vendor_iv)


def build_residual(
    our: ValidationRow,
    *,
    match: Optional[CaptureMatch],
) -> ResidualRow:
    """Classify and compute residuals. ``vendor_missing`` is not disagreement."""
    if our.status is RowStatus.FAILURE:
        return _residual_shell(our, JoinClass.OURS_FAILED, match=None)

    if match is None or not vendor_sane(match.snapshot):
        return _residual_shell(our, JoinClass.VENDOR_MISSING, match=None)

    snap = match.snapshot
    assert our.iv is not None
    assert snap.iv is not None
    mny: Optional[float] = None
    if our.forward is not None and our.forward > 0:
        mny = our.strike / our.forward
    return ResidualRow(
        symbol=our.symbol,
        trade_ts=our.trade_ts,
        methodology_version=our.methodology_version,
        underlying=our.underlying,
        join_class=JoinClass.MATCHED,
        snapshot_ts=snap.timestamp,
        join_lag_capture_ms=match.join_lag_capture_ms,
        quote_ts=snap.quote_timestamp,
        join_lag_quote_ms=match.join_lag_quote_ms,
        our_iv=our.iv,
        our_delta=our.delta,
        our_gamma=our.gamma,
        our_vega=our.vega,
        our_theta=our.theta,
        our_rho=our.rho,
        vendor_iv=snap.iv,
        vendor_delta=snap.delta,
        vendor_gamma=snap.gamma,
        vendor_vega=snap.vega,
        vendor_theta=snap.theta,
        vendor_rho=snap.rho,
        residual_iv_bps=residual_iv_bps(our.iv, snap.iv),
        residual_delta=_sub(our.delta, snap.delta),
        residual_gamma=_sub(our.gamma, snap.gamma),
        residual_vega=_sub(our.vega, snap.vega),
        residual_theta=_sub(our.theta, snap.theta),
        residual_rho=_sub(our.rho, snap.rho),
        moneyness=mny,
        dte_years=our.time_to_expiry,
    )


def join_batch(
    ours: Sequence[ValidationRow],
    snapshots_by_symbol: Mapping[str, Sequence[VendorSnapshot]],
    *,
    max_staleness_s: int = 300,
) -> list[ResidualRow]:
    """Join each validation row to vendor snapshots for its symbol."""
    out: list[ResidualRow] = []
    for row in ours:
        snaps = snapshots_by_symbol.get(row.symbol, ())
        m = match_capture(row.trade_ts, snaps, max_staleness_s=max_staleness_s)
        out.append(build_residual(row, match=m))
    return out


def filter_baseline_snapshots(
    snapshots: Sequence[VendorSnapshot],
    *,
    window_start: date,
    excluded_dates: Sequence[date],
) -> list[VendorSnapshot]:
    """Keep snapshots with capture date ≥ window_start and not excluded."""
    excl = set(excluded_dates)
    out: list[VendorSnapshot] = []
    for s in snapshots:
        d = _aware(s.timestamp).date()
        if d < window_start:
            continue
        if d in excl:
            continue
        out.append(s)
    return out


def _sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def _residual_shell(
    our: ValidationRow,
    join_class: JoinClass,
    *,
    match: Optional[CaptureMatch],
) -> ResidualRow:
    snap = match.snapshot if match else None
    mny: Optional[float] = None
    if our.forward is not None and our.forward > 0:
        mny = our.strike / our.forward
    return ResidualRow(
        symbol=our.symbol,
        trade_ts=our.trade_ts,
        methodology_version=our.methodology_version,
        underlying=our.underlying,
        join_class=join_class,
        snapshot_ts=snap.timestamp if snap else None,
        join_lag_capture_ms=match.join_lag_capture_ms if match else None,
        quote_ts=snap.quote_timestamp if snap else None,
        join_lag_quote_ms=match.join_lag_quote_ms if match else None,
        our_iv=our.iv,
        our_delta=our.delta,
        our_gamma=our.gamma,
        our_vega=our.vega,
        our_theta=our.theta,
        our_rho=our.rho,
        vendor_iv=None,
        vendor_delta=None,
        vendor_gamma=None,
        vendor_vega=None,
        vendor_theta=None,
        vendor_rho=None,
        residual_iv_bps=None,
        residual_delta=None,
        residual_gamma=None,
        residual_vega=None,
        residual_theta=None,
        residual_rho=None,
        moneyness=mny,
        dte_years=our.time_to_expiry,
    )
