"""Per-trade invert: staged trade → carry → scalar IV → ValidationRow.

**Scalar only** (tiers 1–2 path). ``py_vollib_vectorized`` is not used here;
if introduced later for throughput, re-run Tier-1 fixtures bit-for-bit on the
same code path before enabling on live trades (plan PR5 gate).

Every staged trade yields exactly one ``ValidationRow`` (success or failure).
Zero silent drops.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable, Mapping, Optional, Sequence

from greeks.config import GreeksConfig
from greeks.domain import FailureReason, RowStatus, SolverInput
from greeks.forwards.carry import carry
from greeks.forwards.sofr import continuous_rate_from_sofr
from greeks.harness.rows import ValidationRow
from greeks.occ import parse_occ
from greeks.pull.staging import StagedTrade
from greeks.solver.invert import invert


@dataclass(frozen=True)
class InvertBatchStats:
    """Accounting: success + failure must equal input count."""

    n_input: int
    n_success: int
    n_failure: int
    by_reason: Mapping[str, int]

    def balanced(self) -> bool:
        return self.n_input == self.n_success + self.n_failure


def _aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("trade_ts must be timezone-aware")
    return ts.astimezone(timezone.utc)


def invert_staged_trade(
    trade: StagedTrade,
    *,
    cfg: GreeksConfig,
    sofr_series: Mapping[date, float],
    option_price: Optional[float] = None,
) -> ValidationRow:
    """Invert one staged trade. Never returns a default IV on failure.

    ``option_price``: if set, invert this price instead of ``trade.price``
    (e.g. vendor bid/ask mid). The value used is stored on the row as
    ``trade_price``. No silent substitution — caller must pass mid explicitly.
    """
    methodology = cfg.methodology_version
    trade_ts = _aware(trade.trade_ts)
    price = float(option_price) if option_price is not None else float(trade.price)

    try:
        occ = parse_occ(trade.symbol)
    except ValueError:
        return _failure(
            trade,
            methodology,
            FailureReason.NONSTANDARD_CONTRACT,
            expiry=trade.session_date,
            strike=0.0,
            right=None,
        )

    if trade.session_date in cfg.window.excluded_dates:
        return _failure(
            trade,
            methodology,
            FailureReason.EXCLUDED_DATE,
            expiry=occ.expiry,
            strike=occ.strike,
            right=occ.right,
        )

    if trade.spot_at_trade is None or trade.spot_at_trade <= 0:
        return _failure(
            trade,
            methodology,
            FailureReason.MISSING_SPOT,
            expiry=occ.expiry,
            strike=occ.strike,
            right=occ.right,
        )

    try:
        r = continuous_rate_from_sofr(
            sofr_series, trade.session_date, occ.expiry
        )
        carry_out = carry(
            trade.spot_at_trade,
            r,
            trade_ts,
            occ.expiry,
            cfg.dividends_for(trade.underlying),
            expiry_time_et=cfg.expiry_time_et,
        )
    except (KeyError, ValueError):
        return _failure(
            trade,
            methodology,
            FailureReason.MISSING_FORWARD_INPUTS,
            expiry=occ.expiry,
            strike=occ.strike,
            right=occ.right,
            spot=trade.spot_at_trade,
        )

    result = invert(
        SolverInput(
            trade_price=price,
            strike=occ.strike,
            right=occ.right,
            forward=carry_out.forward,
            discount=carry_out.discount,
            time_to_expiry=carry_out.time_to_expiry,
            spot=trade.spot_at_trade,
        ),
        methodology_version=methodology,
        t_floor_minutes=float(cfg.t_floor_minutes),
    )

    if result.status is RowStatus.FAILURE:
        return ValidationRow(
            symbol=trade.symbol,
            trade_ts=trade_ts,
            methodology_version=methodology,
            underlying=trade.underlying.upper(),
            expiry=occ.expiry,
            strike=occ.strike,
            right=occ.right,
            trade_price=price,
            spot_at_trade=trade.spot_at_trade,
            forward=carry_out.forward,
            discount=carry_out.discount,
            time_to_expiry=carry_out.time_to_expiry,
            iv=None,
            delta=None,
            gamma=None,
            vega=None,
            theta=None,
            rho=None,
            status=RowStatus.FAILURE,
            reason_code=result.reason_code,
            theta_interpretively_limited=False,
        )

    return ValidationRow(
        symbol=trade.symbol,
        trade_ts=trade_ts,
        methodology_version=methodology,
        underlying=trade.underlying.upper(),
        expiry=occ.expiry,
        strike=occ.strike,
        right=occ.right,
        trade_price=price,
        spot_at_trade=trade.spot_at_trade,
        forward=carry_out.forward,
        discount=carry_out.discount,
        time_to_expiry=carry_out.time_to_expiry,
        iv=result.iv,
        delta=result.delta,
        gamma=result.gamma,
        vega=result.vega,
        theta=result.theta,
        rho=result.rho,
        status=RowStatus.SUCCESS,
        reason_code=None,
        theta_interpretively_limited=result.theta_interpretively_limited,
    )


def invert_batch(
    trades: Sequence[StagedTrade] | Iterable[StagedTrade],
    *,
    cfg: GreeksConfig,
    sofr_series: Mapping[date, float],
    option_prices: Optional[Sequence[Optional[float]]] = None,
) -> tuple[list[ValidationRow], InvertBatchStats]:
    """Invert all trades; return rows + accounting stats.

    ``option_prices``: optional per-trade prices aligned with ``trades`` (e.g. mids).
    ``None`` entry or omitted list → use each trade's print price. No silent mid.
    """
    trade_list = list(trades)
    if option_prices is not None and len(option_prices) != len(trade_list):
        raise ValueError(
            f"option_prices length {len(option_prices)} != trades {len(trade_list)}"
        )
    rows: list[ValidationRow] = []
    by_reason: dict[str, int] = {}
    n_success = 0
    n_failure = 0
    n = 0
    for i, tr in enumerate(trade_list):
        n += 1
        op: Optional[float] = None
        if option_prices is not None:
            op = option_prices[i]
        row = invert_staged_trade(
            tr, cfg=cfg, sofr_series=sofr_series, option_price=op
        )
        rows.append(row)
        if row.status is RowStatus.SUCCESS:
            n_success += 1
        else:
            n_failure += 1
            key = row.reason_code.value if row.reason_code else "unknown"
            by_reason[key] = by_reason.get(key, 0) + 1
    stats = InvertBatchStats(
        n_input=n,
        n_success=n_success,
        n_failure=n_failure,
        by_reason=by_reason,
    )
    if not stats.balanced():
        raise RuntimeError(
            f"invert accounting broken: input={n} success={n_success} failure={n_failure}"
        )
    return rows, stats


def _failure(
    trade: StagedTrade,
    methodology: str,
    reason: FailureReason,
    *,
    expiry: date,
    strike: float,
    right: Optional[object],
    spot: Optional[float] = None,
) -> ValidationRow:
    from greeks.domain import OptionRight

    r = right if isinstance(right, OptionRight) else OptionRight.CALL
    return ValidationRow(
        symbol=trade.symbol,
        trade_ts=_aware(trade.trade_ts),
        methodology_version=methodology,
        underlying=trade.underlying.upper(),
        expiry=expiry,
        strike=strike,
        right=r,
        trade_price=trade.price,
        spot_at_trade=spot if spot is not None else trade.spot_at_trade,
        forward=None,
        discount=None,
        time_to_expiry=None,
        iv=None,
        delta=None,
        gamma=None,
        vega=None,
        theta=None,
        rho=None,
        status=RowStatus.FAILURE,
        reason_code=reason,
        theta_interpretively_limited=False,
    )
