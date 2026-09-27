"""PR2 — watchlist enumeration: turn the watchlist into queue jobs.

For each watchlist underlying, across the backfill span: one Alpaca **RAW** daily-
bars call gives the trading days and the unadjusted close (the moneyness spot — a
bar exists only on a trading day, so there is no separate calendar and no way to
mistake an API hiccup for a holiday). The Massive contract reference, sampled once
per ISO week, gives what was listed as-of each day. The per-era band selects the
in-band contracts. Every surviving (contract, day) becomes one queue job — so
nothing is ever pulled at PR3 that was not enumerated here (a trade outside the
enumerated set is a hard error at pull time).

Reuses greeks primitives (contract fetch/filter, the RAW adjustment guard). No new
dependencies. The two network calls live in module functions so tests inject fakes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from clickhouse_connect.driver.client import Client

from greeks.pull.alpaca_spot import REQUIRED_ADJUSTMENT, REQUIRED_FEED, assert_raw_adjustment
from greeks.pull.contracts import ContractRef, fetch_massive_contracts, filter_contracts
from option_archive.config import ArchiveConfig
from option_archive.domain import Era, to_osi
from option_archive.queue import WorkQueue

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeedReport:
    """What one seeding pass enumerated. No silent drops — every eligible
    contract-day is either enqueued or already present."""

    underlyings: int
    trading_days: int
    enqueued: int  # newly-inserted queue jobs (existing ones left untouched)
    eligible_contract_days: int
    nonstandard_excluded: int
    days_without_bar: int


def watchlist_underlyings(
    ch: Client, *, table: str, list_name: Optional[str] = None
) -> tuple[str, ...]:
    """Distinct underlyings to seed, from ``trading.watchlist`` (FINAL). With
    ``list_name`` set, only that list; otherwise every symbol on the watchlist."""
    sql = f"SELECT DISTINCT symbol FROM {table} FINAL"
    params: dict[str, str] = {}
    if list_name is not None:
        sql += " WHERE list_name = {ln:String}"
        params["ln"] = list_name
    sql += " ORDER BY symbol"
    result = ch.query(sql, parameters=params)
    return tuple(str(r[0]).upper() for r in result.result_rows)


def era_for(work_date: date, *, now: date, cfg: ArchiveConfig) -> Era:
    """PERISHABLE for the oldest ~year of retained data (closest to the assumed
    roll-off edge, and unrecoverable once gone); ROUTINE otherwise."""
    retention = cfg.roll_off.assumed_retention_years
    perishable_cutoff = now - timedelta(days=365 * (retention - 1))
    return Era.PERISHABLE if work_date <= perishable_cutoff else Era.ROUTINE


def _fetch_daily_raw_closes(
    alpaca: StockHistoricalDataClient, underlying: str, start: date, end: date
) -> dict[date, float]:
    """RAW daily closes keyed by trading day. RAW (unadjusted) is required so
    moneyness is judged against as-traded strikes, never split-adjusted."""
    assert_raw_adjustment()
    req = StockBarsRequest(
        symbol_or_symbols=underlying.upper(),
        timeframe=TimeFrame.Day,
        start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
        end=datetime(end.year, end.month, end.day, tzinfo=timezone.utc),
        adjustment=REQUIRED_ADJUSTMENT,
        feed=REQUIRED_FEED,
    )
    bars = alpaca.get_stock_bars(req)
    data = getattr(bars, "data", {}) or {}
    rows = data.get(underlying.upper()) or data.get(underlying) or []
    out: dict[date, float] = {}
    for bar in rows:
        out[bar.timestamp.date()] = float(bar.close)
    return out


def _week_anchor(d: date) -> date:
    """Monday of ``d``'s ISO week — the as-of date used to sample contracts once
    per week rather than once per day."""
    return d - timedelta(days=d.weekday())


def _contracts_asof(
    api_key: str,
    underlying: str,
    work_date: date,
    max_dte_days: int,
    cache: dict[tuple[str, date, int], list[ContractRef]],
) -> list[ContractRef]:
    key = (underlying, _week_anchor(work_date), max_dte_days)
    if key not in cache:
        cache[key] = fetch_massive_contracts(
            api_key, underlying, as_of=_week_anchor(work_date), max_dte_days=max_dte_days
        )
    return cache[key]


def seed_watchlist(
    queue: WorkQueue,
    ch: Client,
    cfg: ArchiveConfig,
    *,
    massive_api_key: str,
    alpaca: StockHistoricalDataClient,
    now: date,
    list_name: Optional[str] = None,
) -> SeedReport:
    """Enumerate eligible (contract, day) jobs for every watchlist underlying over
    ``[cfg.backfill_start_date, now]`` and enqueue them. Idempotent — re-running
    adds only new jobs (``WorkQueue.enqueue_many`` is INSERT OR IGNORE)."""
    if not massive_api_key:
        raise ValueError("Massive API key is required")
    unders = watchlist_underlyings(ch, table=cfg.tables.watchlist, list_name=list_name)
    start = cfg.backfill_start_date

    trading_days = 0
    eligible = 0
    nonstandard = 0
    days_without_bar = 0
    pairs: list[tuple[str, date]] = []
    contract_cache: dict[tuple[str, date, int], list[ContractRef]] = {}

    for underlying in unders:
        closes = _fetch_daily_raw_closes(alpaca, underlying, start, now)
        # Every calendar day in span with no bar is a non-trading day — expected,
        # counted, never treated as missing data.
        span_days = (now - start).days + 1
        days_without_bar += span_days - len(closes)
        for work_date in sorted(closes):
            if work_date in cfg.excluded_dates:
                continue  # dropped day (collector outage etc.) — never enumerated
            trading_days += 1
            band = cfg.band_for(era_for(work_date, now=now, cfg=cfg))
            contracts = _contracts_asof(
                massive_api_key, underlying, work_date, band.max_dte_days, contract_cache
            )
            result = filter_contracts(
                contracts,
                as_of=work_date,
                spot=closes[work_date],
                max_dte_days=band.max_dte_days,
                moneyness_band=band.moneyness_band,
            )
            nonstandard += len(result.nonstandard)
            for c in result.eligible:
                eligible += 1
                pairs.append((c.osi, work_date))

    enqueued = queue.enqueue_many((to_osi(sym), d) for sym, d in pairs)
    report = SeedReport(
        underlyings=len(unders),
        trading_days=trading_days,
        enqueued=enqueued,
        eligible_contract_days=eligible,
        nonstandard_excluded=nonstandard,
        days_without_bar=days_without_bar,
    )
    log.info(
        "seed_watchlist: unders=%d trading_days=%d eligible=%d enqueued=%d "
        "nonstandard_excluded=%d days_without_bar=%d",
        report.underlyings,
        report.trading_days,
        report.eligible_contract_days,
        report.enqueued,
        report.nonstandard_excluded,
        report.days_without_bar,
    )
    return report
