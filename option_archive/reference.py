"""Watchlist, trading-day, and spot helpers used by ``archive.py``.

For each watchlist underlying, one Alpaca **RAW** daily-bars call gives the trading
days and the unadjusted close (the moneyness spot — a bar exists only on a trading
day, so there is no separate calendar and no way to mistake an API hiccup for a
holiday). Reuses greeks primitives (the RAW adjustment guard). The network calls
live in module functions so tests inject fakes.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from clickhouse_connect.driver.client import Client

from greeks.pull.alpaca_spot import REQUIRED_ADJUSTMENT, REQUIRED_FEED, assert_raw_adjustment
from option_archive.config import ArchiveConfig
from option_archive.domain import Era
from option_archive.retry import with_retry

log = logging.getLogger(__name__)


def watchlist_underlyings(
    ch: Client, *, table: str, list_name: Optional[str] = None
) -> tuple[str, ...]:
    """Distinct underlyings to archive, from ``trading.watchlist`` (FINAL). With
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
    bars = with_retry(lambda: alpaca.get_stock_bars(req), what=f"alpaca daily bars {underlying}")
    data = getattr(bars, "data", {}) or {}
    rows = data.get(underlying.upper()) or data.get(underlying) or []
    out: dict[date, float] = {}
    for bar in rows:
        out[bar.timestamp.date()] = float(bar.close)
    return out


def _week_anchor(d: date) -> date:
    """Monday of ``d``'s ISO week — the as-of date used to sample contracts once per
    week rather than once per day."""
    return d - timedelta(days=d.weekday())
