"""OptionGreeksMessage — the md.option.greeks payload (greeks-server-spec.md §7).

Self-describing (P13). One definition here; at cutover one Scala case class with a derived
upickle codec. tests/greeks_server/option_greeks_golden.json is the cross-language fixture.

Optional fields that have no value are omitted from the JSON — never null, never 0.
  - reason: failure only.  iv/delta/gamma/theta/vega/rho: success only (units §2.1).
  - price/price_kind/price_ts: absent on no_session_price.
  - spot/spot_kind/spot_ts: absent on missing_spot.
  - forward/discount: present whenever the carry was computed.  t_years: whenever T was.

Units: delta spot terms per share; gamma spot terms per share per $1; theta $/share per calendar
day, Black-76 (forward held fixed); vega $/share per vol point; rho $/share per 1.00 rate; iv
annualized decimal. The x100 contract multiplier is applied downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

from greeks.domain import FailureReason, RowStatus
from greeks_server.price import PriceKind

_GREEKS = ("iv", "delta", "gamma", "theta", "vega", "rho")


@dataclass(frozen=True)
class OptionGreeksMessage:
    osi: str
    status: RowStatus
    computed_at: int
    methodology_version: str
    theta_limited: bool = False
    reason: Optional[FailureReason] = None
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    price: Optional[float] = None
    price_kind: Optional[PriceKind] = None
    price_ts: Optional[int] = None
    spot: Optional[float] = None
    spot_kind: Optional[PriceKind] = None
    spot_ts: Optional[int] = None
    forward: Optional[float] = None
    discount: Optional[float] = None
    t_years: Optional[float] = None

    def __post_init__(self) -> None:
        values = [getattr(self, g) for g in _GREEKS]
        if self.status is RowStatus.SUCCESS:
            if self.reason is not None:
                raise ValueError(f"{self.osi}: success must not carry a reason")
            if any(v is None or not math.isfinite(v) for v in values):
                raise ValueError(f"{self.osi}: success requires every greek, finite")
            if self.price is None or self.spot is None or self.forward is None:
                raise ValueError(f"{self.osi}: success requires price, spot and carry")
        else:
            if self.reason is None:
                raise ValueError(f"{self.osi}: failure requires a reason")
            if any(v is not None for v in values):
                raise ValueError(f"{self.osi}: failure must not carry greeks")
        for value, kind, ts, label in (
            (self.price, self.price_kind, self.price_ts, "price"),
            (self.spot, self.spot_kind, self.spot_ts, "spot"),
        ):
            if (value is None) != (kind is None) or (value is None) != (ts is None):
                raise ValueError(f"{self.osi}: {label}, {label}_kind and {label}_ts go together")

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            out[f.name] = v.value if isinstance(v, (RowStatus, FailureReason, PriceKind)) else v
        return out

    @staticmethod
    def from_wire(d: Mapping[str, Any]) -> "OptionGreeksMessage":
        """Strict decode: unknown fields and wrong types are rejected."""
        known = {f.name for f in fields(OptionGreeksMessage)}
        extra = set(d) - known
        if extra:
            raise ValueError(f"unknown fields: {sorted(extra)}")

        def num(key: str) -> Optional[float]:
            v = d.get(key)
            if v is None:
                return None
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{key} must be a number, got {v!r}")
            return float(v)

        def integer(key: str) -> Optional[int]:
            v = d.get(key)
            if v is None:
                return None
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{key} must be an integer, got {v!r}")
            return v

        def kind(key: str) -> Optional[PriceKind]:
            v = d.get(key)
            return None if v is None else PriceKind(v)

        for req in ("osi", "status", "computed_at", "methodology_version", "theta_limited"):
            if req not in d:
                raise ValueError(f"missing field: {req}")
        computed_at = integer("computed_at")
        assert computed_at is not None
        if not isinstance(d["theta_limited"], bool):
            raise ValueError("theta_limited must be a bool")
        return OptionGreeksMessage(
            osi=str(d["osi"]),
            status=RowStatus(d["status"]),
            computed_at=computed_at,
            methodology_version=str(d["methodology_version"]),
            theta_limited=d["theta_limited"],
            reason=None if d.get("reason") is None else FailureReason(d["reason"]),
            iv=num("iv"),
            delta=num("delta"),
            gamma=num("gamma"),
            theta=num("theta"),
            vega=num("vega"),
            rho=num("rho"),
            price=num("price"),
            price_kind=kind("price_kind"),
            price_ts=integer("price_ts"),
            spot=num("spot"),
            spot_kind=kind("spot_kind"),
            spot_ts=integer("spot_ts"),
            forward=num("forward"),
            discount=num("discount"),
            t_years=num("t_years"),
        )
