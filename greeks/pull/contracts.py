"""Enumerate option contracts for a validation as-of date.

Source: Massive ``/v3/reference/options/contracts`` (sync httpx), optional
ClickHouse ``option_contract`` fallback when a client is provided and Massive
is unavailable.

Filters (plan PR4):
- DTE ≤ ``max_dte_days`` (default 90) as of the session date
- moneyness K/S in ``[1 - band, 1 + band]`` (default ±30%)
- standard deliverable only (``shares_per_contract == 100``); nonstandard
  contracts are returned separately with reason ``NONSTANDARD_CONTRACT``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Mapping, Optional, Sequence

import httpx

from greeks.domain import FailureReason, OptionRight
from greeks.occ import OccContract, parse_occ, strip_massive_prefix, to_massive_ticker

MASSIVE_BASE = "https://api.massive.com"
DEFAULT_MAX_DTE_DAYS = 90
DEFAULT_MONEYNESS_BAND = 0.30
STANDARD_DELIVERABLE = 100


@dataclass(frozen=True)
class ContractRef:
    """One listed option contract after parse + eligibility."""

    osi: str
    massive_ticker: str
    underlying: str
    expiry: date
    strike: float
    right: OptionRight
    shares_per_contract: int
    exercise_style: Optional[str]
    primary_exchange: Optional[str]


@dataclass(frozen=True)
class ContractFilterResult:
    """Eligible contracts + nonstandard exclusions (never silent-drop)."""

    eligible: tuple[ContractRef, ...]
    nonstandard: tuple[tuple[ContractRef, FailureReason], ...]
    skipped_dte: int
    skipped_moneyness: int


def dte_days(as_of: date, expiry: date) -> int:
    return (expiry - as_of).days


def moneyness(strike: float, spot: float) -> float:
    if spot <= 0:
        raise ValueError("spot must be > 0 for moneyness")
    return strike / spot


def is_in_moneyness_band(strike: float, spot: float, band: float) -> bool:
    m = moneyness(strike, spot)
    return (1.0 - band) <= m <= (1.0 + band)


def _parse_contract_row(row: Mapping[str, Any], underlying: str) -> Optional[ContractRef]:
    """Map Massive reference row → ContractRef, or None if unparseable OSI."""
    ticker = str(row.get("ticker") or row.get("option_symbol") or "")
    if not ticker:
        return None
    try:
        occ = parse_occ(ticker)
    except ValueError:
        return None
    shares = row.get("shares_per_contract")
    if shares is None:
        shares = row.get("contract_size", STANDARD_DELIVERABLE)
    try:
        shares_i = int(shares)
    except (TypeError, ValueError):
        shares_i = -1
    exp_raw = row.get("expiration_date")
    if exp_raw is not None:
        exp = date.fromisoformat(str(exp_raw)[:10])
    else:
        exp = occ.expiry
    strike = float(row.get("strike_price", occ.strike))
    ct = str(row.get("contract_type", "")).lower()
    if ct.startswith("c"):
        right = OptionRight.CALL
    elif ct.startswith("p"):
        right = OptionRight.PUT
    else:
        right = occ.right
    return ContractRef(
        osi=occ.symbol,
        massive_ticker=to_massive_ticker(occ.symbol),
        underlying=underlying.upper(),
        expiry=exp,
        strike=strike,
        right=right,
        shares_per_contract=shares_i,
        exercise_style=(
            str(row["exercise_style"]) if row.get("exercise_style") is not None else None
        ),
        primary_exchange=(
            str(row["primary_exchange"])
            if row.get("primary_exchange") is not None
            else None
        ),
    )


def filter_contracts(
    contracts: Sequence[ContractRef],
    *,
    as_of: date,
    spot: float,
    max_dte_days: int = DEFAULT_MAX_DTE_DAYS,
    min_dte_days: int = 0,
    moneyness_band: float = DEFAULT_MONEYNESS_BAND,
    standard_deliverable: int = STANDARD_DELIVERABLE,
) -> ContractFilterResult:
    """Apply DTE, moneyness, and deliverable filters."""
    if min_dte_days < 0:
        raise ValueError("min_dte_days must be >= 0")
    if min_dte_days > max_dte_days:
        raise ValueError("min_dte_days must be <= max_dte_days")
    eligible: list[ContractRef] = []
    nonstandard: list[tuple[ContractRef, FailureReason]] = []
    skipped_dte = 0
    skipped_moneyness = 0
    for c in contracts:
        if c.shares_per_contract != standard_deliverable:
            nonstandard.append((c, FailureReason.NONSTANDARD_CONTRACT))
            continue
        dte = dte_days(as_of, c.expiry)
        if dte < min_dte_days or dte > max_dte_days:
            skipped_dte += 1
            continue
        if not is_in_moneyness_band(c.strike, spot, moneyness_band):
            skipped_moneyness += 1
            continue
        eligible.append(c)
    return ContractFilterResult(
        eligible=tuple(eligible),
        nonstandard=tuple(nonstandard),
        skipped_dte=skipped_dte,
        skipped_moneyness=skipped_moneyness,
    )


def fetch_massive_contracts(
    api_key: str,
    underlying: str,
    *,
    as_of: date,
    max_dte_days: int = DEFAULT_MAX_DTE_DAYS,
    expired: bool = True,
    timeout_s: float = 60.0,
    client: Optional[httpx.Client] = None,
) -> list[ContractRef]:
    """Page Massive reference contracts for ``underlying``.

    Requests expiration window ``[as_of, as_of + max_dte_days]``. When
    ``expired=True``, includes contracts that may have expired after ``as_of``
    (needed for historical validation days).
    """
    if not api_key:
        raise ValueError("Massive API key is required")
    und = underlying.upper()
    exp_gte = as_of.isoformat()
    exp_lte = (as_of + timedelta(days=max_dte_days)).isoformat()
    params: dict[str, str] = {
        "underlying_ticker": und,
        "expiration_date.gte": exp_gte,
        "expiration_date.lte": exp_lte,
        "limit": "1000",
        "sort": "expiration_date",
        "order": "asc",
        "apiKey": api_key,
    }
    # Historical days need expired contracts; live discovery uses expired=false.
    if expired:
        params["expired"] = "true"
    else:
        params["expired"] = "false"

    own_client = client is None
    http = client or httpx.Client(timeout=timeout_s)
    out: list[ContractRef] = []
    try:
        url: Optional[str] = f"{MASSIVE_BASE}/v3/reference/options/contracts"
        next_url: Optional[str] = None
        while True:
            if next_url:
                sep = "&" if "?" in next_url else "?"
                resp = http.get(f"{next_url}{sep}apiKey={api_key}")
            else:
                assert url is not None
                resp = http.get(url, params=params)
            resp.raise_for_status()
            body = resp.json()
            results = body.get("results") or []
            if not isinstance(results, list):
                raise ValueError("Massive contracts response missing results list")
            for row in results:
                if isinstance(row, Mapping):
                    ref = _parse_contract_row(row, und)
                    if ref is not None:
                        out.append(ref)
            next_url = body.get("next_url")
            if not next_url:
                break
    finally:
        if own_client:
            http.close()
    return out


def contracts_from_clickhouse_rows(
    rows: Sequence[Mapping[str, Any]],
    underlying: str,
) -> list[ContractRef]:
    """Map CH ``option_contract``-like rows to ContractRef."""
    und = underlying.upper()
    out: list[ContractRef] = []
    for row in rows:
        sym = str(row.get("option_symbol") or row.get("symbol") or "")
        if not sym:
            continue
        try:
            occ = parse_occ(sym)
        except ValueError:
            continue
        size = int(row.get("contract_size", STANDARD_DELIVERABLE))
        exp = row.get("expiration_date")
        if isinstance(exp, date):
            expiry = exp
        else:
            expiry = date.fromisoformat(str(exp)[:10]) if exp else occ.expiry
        strike = float(row.get("strike_price", occ.strike))
        cp = str(row.get("call_put", occ.right.value)).upper()
        right = OptionRight.CALL if cp.startswith("C") else OptionRight.PUT
        out.append(
            ContractRef(
                osi=strip_massive_prefix(sym),
                massive_ticker=to_massive_ticker(sym),
                underlying=und,
                expiry=expiry,
                strike=strike,
                right=right,
                shares_per_contract=size,
                exercise_style=None,
                primary_exchange=None,
            )
        )
    return out


def occ_from_ref(ref: ContractRef) -> OccContract:
    return OccContract(
        symbol=ref.osi,
        root=ref.underlying,
        expiry=ref.expiry,
        right=ref.right,
        strike=ref.strike,
    )
