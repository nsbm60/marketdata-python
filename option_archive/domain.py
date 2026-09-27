"""Frozen domain types and enums for the option_archive package.

Mirrors ``greeks/domain.py``: no silent defaults, enums for every status/source,
frozen dataclasses. Reuses ``OptionRight`` from the greeks package rather than
redefining it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import NewType, Optional

# Reused, not redefined — the archive speaks the same option vocabulary as greeks.
from greeks.domain import OptionRight
from greeks.occ import parse_occ

# An OSI option symbol, distinct from an arbitrary str. The only way to make one
# is to_osi(), which validates — so a contract that reaches the queue is a
# provably well-formed symbol, not a hopeful string.
OsiSymbol = NewType("OsiSymbol", str)


def to_osi(symbol: str) -> OsiSymbol:
    """Validate a standard OSI symbol at the entry point; raise on a garbled one.

    Fast-fail: a bad contract identifier never enters the queue. (A non-standard
    *deliverable* — shares_per_contract != 100 — is a separate concern handled by
    the enumeration filter with a reason code, not by this format check.)
    """
    parse_occ(symbol)  # raises ValueError if not a standard OSI
    return OsiSymbol(symbol)


__all__ = [
    "OptionRight",
    "OsiSymbol",
    "to_osi",
    "TaskStatus",
    "Era",
    "ScheduleWindowKind",
    "TaskFailure",
    "BandSpec",
    "WorkTask",
]


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class ScheduleWindowKind(str, Enum):
    """Which pacing window a schedule entry is: overnight full-throttle vs
    daytime reduced-rate. A closed set, so an enum — not a free-text name."""

    AGGRESSIVE = "aggressive"
    POLITE = "polite"


class Era(str, Enum):
    """Capture era for a (contract, date), which selects the band.

    Perishable = within a year of the assumed roll-off edge at queue-build time
    (err wide, it can never be re-pulled). Routine = re-pullable if regretted.
    """

    PERISHABLE = "perishable"
    ROUTINE = "routine"


class TaskFailure(str, Enum):
    """Reason codes for a task that did not produce success rows.

    Every non-success task carries one of these — never a silent drop. Transport
    and connectivity failures are deliberately ABSENT: an outage returns the task
    to ``PENDING`` without touching ``attempt_count`` (PR1 resilience), so it is
    not a failure reason.
    """

    VENDOR_ERROR = "vendor_error"                # increments attempt_count
    NO_TRADES = "no_trades"                       # terminal, reasoned marker — not a fault
    NONSTANDARD_CONTRACT = "nonstandard_contract"  # deliverable != 100 / adjusted
    ENUMERATION_MISS = "enumeration_miss"         # trade outside enumerated set — hard error


@dataclass(frozen=True)
class BandSpec:
    """Per-era capture band: moneyness half-width and DTE ceiling.

    ``moneyness_band`` is the half-width around 1.0 (0.30 = strikes within
    ±30% of spot). Both are required and validated — a zero or negative band is a
    config error, not a silently-empty capture.
    """

    moneyness_band: float
    max_dte_days: int

    def __post_init__(self) -> None:
        if not (0.0 < self.moneyness_band <= 1.0):
            raise ValueError(
                f"moneyness_band must be in (0, 1], got {self.moneyness_band!r}"
            )
        if self.max_dte_days <= 0:
            raise ValueError(f"max_dte_days must be > 0, got {self.max_dte_days!r}")


@dataclass(frozen=True)
class WorkTask:
    """One queue row: a (contract, date) unit of pull work.

    A single task pulls the contract-day's trades AND (once quotes activate) its
    quotes, joins locally, and inserts complete rows once — there is no separate
    quote task and no second-phase rewrite. ``batch_id`` / ``claimed_at`` are
    populated by the queue (PR1); ``claimed_at`` backs the lease that reverts a
    dead worker's claim to pending.
    """

    contract: OsiSymbol  # validated at enqueue via to_osi — never a bare str
    work_date: date
    status: TaskStatus
    attempt_count: int
    # Fencing token of the claim that produced this WorkTask. Every queue mutation
    # is conditioned on it — if a lease reclaim superseded the claim, the token no
    # longer matches and the stale worker's verdict is a logged no-op, not applied.
    claim_id: Optional[str] = None
    batch_id: Optional[str] = None
    claimed_at: Optional[datetime] = None
    last_error: Optional[str] = None
