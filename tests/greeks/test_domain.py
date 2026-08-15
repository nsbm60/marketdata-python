"""PR0: domain invariants for SolverResult and enums."""

from __future__ import annotations

import pytest

from greeks.domain import FailureReason, OptionRight, RowStatus, SolverInput, SolverResult


def test_success_requires_iv() -> None:
    with pytest.raises(ValueError, match="requires iv"):
        SolverResult(status=RowStatus.SUCCESS, methodology_version="v1")


def test_success_rejects_reason_code() -> None:
    with pytest.raises(ValueError, match="must not carry reason_code"):
        SolverResult(
            status=RowStatus.SUCCESS,
            methodology_version="v1",
            iv=0.2,
            reason_code=FailureReason.SOLVER_FAILED,
        )


def test_failure_requires_reason_code() -> None:
    with pytest.raises(ValueError, match="requires reason_code"):
        SolverResult(status=RowStatus.FAILURE, methodology_version="v1")


def test_failure_rejects_iv() -> None:
    with pytest.raises(ValueError, match="must not carry iv"):
        SolverResult(
            status=RowStatus.FAILURE,
            methodology_version="v1",
            reason_code=FailureReason.BELOW_INTRINSIC,
            iv=0.1,
        )


def test_valid_success_and_failure() -> None:
    ok = SolverResult(status=RowStatus.SUCCESS, methodology_version="v1", iv=0.25)
    assert ok.reason_code is None
    bad = SolverResult(
        status=RowStatus.FAILURE,
        methodology_version="v1",
        reason_code=FailureReason.T_BELOW_FLOOR,
    )
    assert bad.iv is None


def test_solver_input_frozen() -> None:
    inp = SolverInput(
        trade_price=1.0,
        strike=100.0,
        right=OptionRight.CALL,
        forward=100.0,
        discount=0.99,
        time_to_expiry=0.1,
    )
    with pytest.raises(Exception):
        inp.trade_price = 2.0  # type: ignore[misc]
