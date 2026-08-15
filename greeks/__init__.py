"""Greeks validation package (Black-76 + py_vollib methodology gate).

Spec: Scala MarketData docs/investigations/greeks-validation-brief.md
Plan: Scala MarketData docs/plans/greeks-validation.md
"""

from greeks.config import GreeksConfig, get_config, load_config
from greeks.domain import (
    FailureReason,
    JoinClass,
    OptionRight,
    RowStatus,
    SolverInput,
    SolverResult,
)

__all__ = [
    "FailureReason",
    "GreeksConfig",
    "JoinClass",
    "OptionRight",
    "RowStatus",
    "SolverInput",
    "SolverResult",
    "get_config",
    "load_config",
]

__version__ = "0.1.0"
