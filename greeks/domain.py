"""Frozen domain types and enums for the greeks validation package.

No silent defaults. Failures are reason-coded; statuses are explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional


class RowStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class FailureReason(str, Enum):
    """Distinct reason codes for inversion / eligibility failures.

    Every failure path writes a failure row with one of these — never a default IV.
    """

    BELOW_INTRINSIC = "below_intrinsic"
    ABOVE_NO_ARB = "above_no_arb"
    T_BELOW_FLOOR = "t_below_floor"
    QUANTIZED_NEAR_INTRINSIC = "quantized_near_intrinsic"
    SOLVER_FAILED = "solver_failed"
    NONSTANDARD_CONTRACT = "nonstandard_contract"
    MISSING_SPOT = "missing_spot"
    MISSING_FORWARD_INPUTS = "missing_forward_inputs"
    EXCLUDED_DATE = "excluded_date"
    # Diagnostic: --price mid requires a two-sided vendor quote at join; not a solver fault.
    MISSING_VENDOR_MID = "missing_vendor_mid"


class JoinClass(str, Enum):
    """Harness join outcome vs vendor option_snapshot."""

    MATCHED = "matched"
    VENDOR_MISSING = "vendor_missing"
    OURS_FAILED = "ours_failed"


class OptionRight(str, Enum):
    CALL = "C"
    PUT = "P"


class WorkStatus(str, Enum):
    """SQLite work-queue status (PR4)."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Dividend:
    """Single discrete cash dividend on the underlying."""

    underlying: str
    amount: float
    ex_date: date


@dataclass(frozen=True)
class SolverInput:
    """Inputs required for Black-76 IV inversion (one trade)."""

    trade_price: float
    strike: float
    right: OptionRight
    forward: float
    discount: float
    time_to_expiry: float
    spot: Optional[float] = None  # required when converting delta to spot terms


@dataclass(frozen=True)
class SolverResult:
    """Successful inversion + analytic greeks, or a failure reason."""

    status: RowStatus
    methodology_version: str
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    reason_code: Optional[FailureReason] = None
    theta_interpretively_limited: bool = False

    def __post_init__(self) -> None:
        if self.status is RowStatus.SUCCESS:
            if self.reason_code is not None:
                raise ValueError("SUCCESS result must not carry reason_code")
            if self.iv is None:
                raise ValueError("SUCCESS result requires iv")
        elif self.status is RowStatus.FAILURE:
            if self.reason_code is None:
                raise ValueError("FAILURE result requires reason_code")
            if self.iv is not None:
                raise ValueError("FAILURE result must not carry iv")


@dataclass(frozen=True)
class ValidationWindow:
    """Inclusive calendar window and excluded session dates."""

    start: date
    end: date
    excluded_dates: tuple[date, ...]

    def contains(self, d: date) -> bool:
        if d in self.excluded_dates:
            return False
        return self.start <= d <= self.end


@dataclass(frozen=True)
class TradeKey:
    """Identity of a captured option trade for validation rows."""

    symbol: str
    trade_ts: datetime  # UTC
    underlying: str
