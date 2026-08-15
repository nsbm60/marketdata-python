"""Frozen row types for validation and residual outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from greeks.domain import FailureReason, JoinClass, OptionRight, RowStatus


@dataclass(frozen=True)
class ValidationRow:
    """One row for ``trading.greeks_validation`` (success or failure)."""

    symbol: str
    trade_ts: datetime
    methodology_version: str
    underlying: str
    expiry: date
    strike: float
    right: OptionRight
    trade_price: float
    spot_at_trade: Optional[float]
    forward: Optional[float]
    discount: Optional[float]
    time_to_expiry: Optional[float]
    iv: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    vega: Optional[float]
    theta: Optional[float]
    rho: Optional[float]
    status: RowStatus
    reason_code: Optional[FailureReason]
    theta_interpretively_limited: bool = False


@dataclass(frozen=True)
class VendorSnapshot:
    """Subset of ``option_snapshot`` needed for join (capture + quote clocks)."""

    symbol: str
    underlying: str
    timestamp: datetime  # capture clock
    quote_timestamp: Optional[datetime]
    bid: Optional[float]
    ask: Optional[float]
    iv: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    vega: Optional[float]
    theta: Optional[float]
    rho: Optional[float]


@dataclass(frozen=True)
class ResidualRow:
    """One row for ``trading.greeks_residuals``."""

    symbol: str
    trade_ts: datetime
    methodology_version: str
    underlying: str
    join_class: JoinClass
    snapshot_ts: Optional[datetime]
    join_lag_capture_ms: Optional[int]
    quote_ts: Optional[datetime]
    join_lag_quote_ms: Optional[int]
    our_iv: Optional[float]
    our_delta: Optional[float]
    our_gamma: Optional[float]
    our_vega: Optional[float]
    our_theta: Optional[float]
    our_rho: Optional[float]
    vendor_iv: Optional[float]
    vendor_delta: Optional[float]
    vendor_gamma: Optional[float]
    vendor_vega: Optional[float]
    vendor_theta: Optional[float]
    vendor_rho: Optional[float]
    residual_iv_bps: Optional[float]
    residual_delta: Optional[float]
    residual_gamma: Optional[float]
    residual_vega: Optional[float]
    residual_theta: Optional[float]
    residual_rho: Optional[float]
    moneyness: Optional[float]
    dte_years: Optional[float]
