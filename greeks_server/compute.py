"""Price one contract (greeks-server-spec.md §2, §5.4, §7). Pure given its inputs.

No pricing math of its own: carry from greeks.forwards.carry, T from greeks.solver.time, IV and
greeks from greeks.solver.invert (through the iv_for seam).
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from greeks.domain import FailureReason, OptionRight, RowStatus, SolverInput, SolverResult
from greeks.forwards.carry import carry
from greeks.occ import OccContract, parse_occ
from greeks.solver.invert import invert
from greeks_server.inputs import SessionInputs
from greeks_server.message import OptionGreeksMessage
from greeks_server.price import SelectedPrice, current_option_mid, current_spot_mid
from greeks_server.wire import Quote

log = logging.getLogger(__name__)


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
    priced_as_of_ms: int,
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
        priced_as_of=priced_as_of_ms,
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
    as_of: datetime,
    m: Methodology,
) -> OptionGreeksMessage:
    """Price one contract as of the moment ``as_of`` (= T). Time to expiry runs from as_of; the
    rate and dividends come from inputs loaded for as_of's date."""
    priced_as_of_ms = int(as_of.timestamp() * 1000)
    # Option first: with no current two-sided quote there is no market to price, and no stock is
    # needed (the cold path doesn't even look one up).
    if option_px is None:
        return _failure(contract, FailureReason.NO_CURRENT_QUOTE, priced_as_of_ms, m, None, spot_px)
    if spot_px is None:
        return _failure(contract, FailureReason.MISSING_SPOT, priced_as_of_ms, m, option_px, None)

    if contract.expiry < inputs.session_date:
        # Expired before as_of's date: there is no rate for a negative term. The solver's floor
        # (t_below_floor) is the reason it would give; the floor itself stays the solver's.
        return _failure(contract, FailureReason.T_BELOW_FLOOR, priced_as_of_ms, m, option_px, spot_px)

    r = inputs.rate_for(contract.expiry)
    c = carry(
        spot_px.value, r, as_of, contract.expiry,
        inputs.dividends_for(contract.underlying), expiry_time_et=m.expiry_time_et,
    )
    res = iv_for(contract, option_px.value, c.forward, c.discount, c.time_to_expiry, spot_px.value, m)
    if res.status is not RowStatus.SUCCESS:
        assert res.reason_code is not None  # SolverResult invariant
        return _failure(
            contract, res.reason_code, priced_as_of_ms, m, option_px, spot_px,
            forward=c.forward, discount=c.discount, t=c.time_to_expiry,
        )
    return OptionGreeksMessage(
        osi=contract.osi,
        status=RowStatus.SUCCESS,
        priced_as_of=priced_as_of_ms,
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


def price_current(
    contract: Contract,
    opt_quote: Optional[Quote],
    und_quote: Optional[Quote],
    as_of: datetime,
    session_open_ms: int,
    inputs: SessionInputs,
    m: Methodology,
) -> OptionGreeksMessage:
    """The one pricing rule the cold path and compute_greeks share, so it lives in one place.

    Prices ``contract`` as of ``as_of``. The option price is the option quote's mid, valid only if
    two-sided, stamped at or before as_of, and at or after ``session_open_ms`` + 1 minute — else
    no_current_quote. The spot is the und quote's mid, valid only if two-sided and at or before
    as_of — else missing_spot. Never a trade. Rate and dividends come from ``inputs`` (as_of's date).
    """
    as_of_ms = int(as_of.timestamp() * 1000)
    opt_px = current_option_mid(opt_quote, as_of_ms, session_open_ms)
    spot_px = current_spot_mid(und_quote, as_of_ms)
    return compute_or_fail(contract, opt_px, spot_px, inputs, as_of, m)


def compute_or_fail(
    contract: Contract,
    option_px: Optional[SelectedPrice],
    spot_px: Optional[SelectedPrice],
    inputs: SessionInputs,
    as_of: datetime,
    m: Methodology,
) -> OptionGreeksMessage:
    """compute(), but one contract can't take the server down: an unexpected error (bad dividend
    row, a non-finite greek from the library) is published as solver_failed for that contract."""
    try:
        return compute(contract, option_px, spot_px, inputs, as_of, m)
    except Exception as e:  # noqa: BLE001 — per-contract isolation, logged
        log.warning("%s: compute raised %s: %s — publishing solver_failed", contract.osi, type(e).__name__, e)
        return _failure(
            contract, FailureReason.SOLVER_FAILED, int(as_of.timestamp() * 1000), m, option_px, spot_px
        )
