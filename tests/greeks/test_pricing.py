"""The shared pricing core: price_options (batch) must equal price_option called per row, across a
mixed batch — successes, every failure reason including expiry-before-session, and a carry-raising row.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from greeks.domain import Dividend, FailureReason, OptionRight
from greeks.pricing import OptionInput, price_option, price_options

SESSION = date(2026, 10, 20)
MV, TFLOOR, ETIME = "v2", 15.0, "16:00"
NOW = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)      # inside the session
NEAR_EXPIRY = datetime(2026, 10, 20, 19, 50, tzinfo=timezone.utc)  # 15:50 ET, 10 min to 16:00 ET expiry
SHORT_DATED = datetime(2026, 10, 20, 14, 0, tzinfo=timezone.utc)   # 10:00 ET, ~6h to expiry (< 1 day)
_DIVS = (Dividend("NVDA", 0.25, date(2026, 11, 5)),)


def _rate_for(_expiry: date) -> float:
    return 0.04


def _oi(expiry: date, right: OptionRight, strike: float, price: float, spot: float,
        as_of: datetime) -> OptionInput:
    return OptionInput(expiry=expiry, right=right, strike=strike, price=price, spot=spot,
                       as_of=as_of, rate_for=_rate_for, dividends=_DIVS)


def test_price_options_equals_price_option_per_row() -> None:
    inputs = [
        _oi(date(2026, 11, 20), OptionRight.CALL, 180.0, 9.1, 181.0, NOW),      # success
        _oi(date(2026, 11, 20), OptionRight.PUT, 180.0, 8.0, 181.0, NOW),       # success
        _oi(date(2026, 1, 1), OptionRight.CALL, 180.0, 9.1, 181.0, NOW),        # expiry<session -> T_BELOW_FLOOR, no carry
        _oi(date(2026, 10, 20), OptionRight.CALL, 180.0, 1.5, 181.0, NEAR_EXPIRY),  # t<floor -> T_BELOW_FLOOR, carry set
        _oi(date(2026, 11, 20), OptionRight.CALL, 100.0, 1.0, 181.0, NOW),      # below_intrinsic
        _oi(date(2026, 11, 20), OptionRight.CALL, 180.0, 1e6, 181.0, NOW),      # above_no_arb
        _oi(date(2026, 10, 20), OptionRight.CALL, 200.0, 0.005, 181.0, SHORT_DATED),  # quantized_near_intrinsic
        _oi(date(2026, 11, 20), OptionRight.CALL, 180.0, 9.1, 0.0, NOW),        # carry raises -> SOLVER_FAILED
    ]
    batch = price_options(inputs, session_date=SESSION, methodology_version=MV,
                          t_floor_minutes=TFLOOR, expiry_time_et=ETIME)
    per_row = [
        price_option(expiry=inp.expiry, right=inp.right, strike=inp.strike, price=inp.price,
                     spot=inp.spot, as_of=inp.as_of, session_date=SESSION, rate_for=inp.rate_for,
                     dividends=inp.dividends, methodology_version=MV, t_floor_minutes=TFLOOR,
                     expiry_time_et=ETIME)
        for inp in inputs
    ]
    assert batch == per_row  # frozen PricedGreeks, bit-for-bit (n=1 wrapper is the same code path)

    reasons = [p.reason for p in batch]
    assert reasons[0] is None and reasons[1] is None
    assert reasons[2] is FailureReason.T_BELOW_FLOOR and batch[2].forward is None   # expiry: no carry
    assert reasons[3] is FailureReason.T_BELOW_FLOOR and batch[3].forward is not None  # floor: carry set
    assert reasons[4] is FailureReason.BELOW_INTRINSIC and batch[4].forward is not None
    assert reasons[5] is FailureReason.ABOVE_NO_ARB
    assert reasons[6] is FailureReason.QUANTIZED_NEAR_INTRINSIC
    assert reasons[7] is FailureReason.SOLVER_FAILED and batch[7].forward is None   # carry raised
