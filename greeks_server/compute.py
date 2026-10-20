"""Price one contract (greeks-server-spec.md §2, §5.4, §7). Pure given its inputs.

No pricing math of its own: carry from greeks.forwards.carry, T from greeks.solver.time, IV and
greeks from greeks.solver.invert (through the iv_for seam).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from greeks.domain import FailureReason, OptionRight, RowStatus, SolverInput, SolverResult
from greeks.forwards.carry import carry
from greeks.occ import OccContract, parse_occ
from greeks.solver.invert import invert
from greeks.solver.time import years_to_expiry
from greeks_server.inputs import SessionInputs
from greeks_server.message import OptionGreeksMessage
from greeks_server.price import SelectedPrice


@dataclass(frozen=True)
class Contract:
    osi: str
    underlying: str
    expiry: date
    right: OptionRight
    strike: float

    @staticmethod
    def parse(osi: str) -> "Contract":
        """Standard OSI only; raises ValueError otherwise (adjusted contracts aren't priced)."""
        c: OccContract = parse_occ(osi)
        return Contract(osi=c.symbol, underlying=c.root, expiry=c.expiry, right=c.right, strike=c.strike)

    @property
    def side(self) -> str:
        return self.right.value  # 'C' / 'P', the topic's side char


@dataclass(frozen=True)
class Methodology:
    methodology_version: str
    t_floor_minutes: float
    expiry_time_et: str


def iv_for(
    contract: Contract,
    price: float,
    forward: float,
    discount: float,
    t: float,
    spot: float,
    m: Methodology,
) -> SolverResult:
    """The IV-source seam. v1: invert the market price. A vol model replaces this and nothing else."""
    return invert(
        SolverInput(
            trade_price=price,
            strike=contract.strike,
            right=contract.right,
            forward=forward,
            discount=discount,
            time_to_expiry=t,
            spot=spot,
        ),
        methodology_version=m.methodology_version,
        t_floor_minutes=m.t_floor_minutes,
    )


def _failure(
    contract: Contract,
    reason: FailureReason,
    now_ms: int,
    m: Methodology,
    option_px: Optional[SelectedPrice],
    spot_px: Optional[SelectedPrice],
    forward: Optional[float] = None,
    discount: Optional[float] = None,
    t: Optional[float] = None,
) -> OptionGreeksMessage:
    return OptionGreeksMessage(
        osi=contract.osi,
        status=RowStatus.FAILURE,
        reason=reason,
        computed_at=now_ms,
        methodology_version=m.methodology_version,
        price=None if option_px is None else option_px.value,
        price_kind=None if option_px is None else option_px.kind,
        price_ts=None if option_px is None else option_px.ts_ms,
        spot=None if spot_px is None else spot_px.value,
        spot_kind=None if spot_px is None else spot_px.kind,
        spot_ts=None if spot_px is None else spot_px.ts_ms,
        forward=forward,
        discount=discount,
        t_years=t,
    )


def compute(
    contract: Contract,
    option_px: Optional[SelectedPrice],
    spot_px: Optional[SelectedPrice],
    inputs: SessionInputs,
    now_utc: datetime,
    m: Methodology,
) -> OptionGreeksMessage:
    now_ms = int(now_utc.timestamp() * 1000)
    if spot_px is None:
        return _failure(contract, FailureReason.MISSING_SPOT, now_ms, m, option_px, None)
    if option_px is None:
        return _failure(contract, FailureReason.NO_SESSION_PRICE, now_ms, m, None, spot_px)

    if contract.expiry < inputs.session_date:
        # Expired before this session: there is no rate for a negative term. The solver's floor
        # (t_below_floor) is the reason it would give; the floor itself stays the solver's.
        t = years_to_expiry(now_utc, contract.expiry, expiry_time_et=m.expiry_time_et)
        return _failure(contract, FailureReason.T_BELOW_FLOOR, now_ms, m, option_px, spot_px, t=t)

    r = inputs.rate_for(contract.expiry)
    c = carry(
        spot_px.value, r, now_utc, contract.expiry,
        inputs.dividends_for(contract.underlying), expiry_time_et=m.expiry_time_et,
    )
    res = iv_for(contract, option_px.value, c.forward, c.discount, c.time_to_expiry, spot_px.value, m)
    if res.status is not RowStatus.SUCCESS:
        assert res.reason_code is not None  # SolverResult invariant
        return _failure(
            contract, res.reason_code, now_ms, m, option_px, spot_px,
            forward=c.forward, discount=c.discount, t=c.time_to_expiry,
        )
    return OptionGreeksMessage(
        osi=contract.osi,
        status=RowStatus.SUCCESS,
        computed_at=now_ms,
        methodology_version=m.methodology_version,
        theta_limited=res.theta_interpretively_limited,
        iv=res.iv,
        delta=res.delta,
        gamma=res.gamma,
        theta=res.theta,
        vega=res.vega,
        rho=res.rho,
        price=option_px.value,
        price_kind=option_px.kind,
        price_ts=option_px.ts_ms,
        spot=spot_px.value,
        spot_kind=spot_px.kind,
        spot_ts=spot_px.ts_ms,
        forward=c.forward,
        discount=c.discount,
        t_years=c.time_to_expiry,
    )
