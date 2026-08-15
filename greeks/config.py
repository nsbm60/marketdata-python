"""Sole config-parsing chokepoint for the greeks package.

All configuration is loaded through :func:`load_config`. Other modules must not
read env vars or YAML for greeks settings except via the returned
:class:`GreeksConfig` (or a value derived from it and passed in).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from greeks.domain import Dividend, ValidationWindow

# Repo-root config path (marketdata-python/config/greeks.yaml)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "greeks.yaml"


@dataclass(frozen=True)
class TableNames:
    greeks_validation: str
    greeks_residuals: str
    option_snapshot: str
    option_contract: str
    earnings_calendar: str
    sofr_daily: str


@dataclass(frozen=True)
class ClickHouseSettings:
    """Connection settings resolved once at config load (env overrides allowed)."""

    host: Optional[str]
    port: int
    user: str
    password: str
    database: str


@dataclass(frozen=True)
class ApiKeys:
    """Vendor credentials from env — empty string if unset (callers must check)."""

    massive_api_key: str
    alpaca_api_key: str
    alpaca_api_secret: str
    fred_api_key: str


@dataclass(frozen=True)
class GreeksConfig:
    """Immutable application config for greeks validation."""

    methodology_version: str
    window: ValidationWindow
    tickers: tuple[str, ...]
    t_floor_minutes: int
    join_staleness_s: int
    expiry_time_et: str
    day_count: str
    sofr_table: str
    dividends: tuple[Dividend, ...]
    tables: TableNames
    clickhouse: ClickHouseSettings
    api_keys: ApiKeys
    config_path: Path

    def dividends_for(self, underlying: str) -> tuple[Dividend, ...]:
        u = underlying.upper()
        return tuple(d for d in self.dividends if d.underlying == u)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _require_mapping(raw: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"config key '{key}' must be a mapping")
    return raw


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"greeks config not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise TypeError(f"greeks config root must be a mapping: {path}")
    return data


def _parse_dividends(raw: Mapping[str, Any]) -> tuple[Dividend, ...]:
    out: list[Dividend] = []
    for underlying, entries in raw.items():
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise TypeError(f"dividends.{underlying} must be a list")
        for item in entries:
            if not isinstance(item, Mapping):
                raise TypeError(f"dividends.{underlying} entries must be mappings")
            amount = item.get("amount")
            ex = item.get("ex_date")
            if amount is None or ex is None:
                raise ValueError(
                    f"dividends.{underlying} entry requires amount and ex_date"
                )
            out.append(
                Dividend(
                    underlying=str(underlying).upper(),
                    amount=float(amount),
                    ex_date=_parse_date(str(ex)),
                )
            )
    return tuple(out)


def _parse_tables(raw: Mapping[str, Any], sofr_table: str) -> TableNames:
    return TableNames(
        greeks_validation=str(raw.get("greeks_validation", "trading.greeks_validation")),
        greeks_residuals=str(raw.get("greeks_residuals", "trading.greeks_residuals")),
        option_snapshot=str(raw.get("option_snapshot", "trading.option_snapshot")),
        option_contract=str(raw.get("option_contract", "trading.option_contract")),
        earnings_calendar=str(raw.get("earnings_calendar", "trading.earnings_calendar")),
        sofr_daily=str(raw.get("sofr_daily", sofr_table)),
    )


def _clickhouse_from_env() -> ClickHouseSettings:
    host = os.environ.get("CLICKHOUSE_HOST")
    port_s = os.environ.get("CLICKHOUSE_PORT", "8123")
    return ClickHouseSettings(
        host=host if host else None,
        port=int(port_s),
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
        database=os.environ.get("CLICKHOUSE_DATABASE", "trading"),
    )


def _api_keys_from_env() -> ApiKeys:
    return ApiKeys(
        massive_api_key=os.environ.get("MASSIVE_API_KEY", "")
        or os.environ.get("POLYGON_API_KEY", ""),
        alpaca_api_key=os.environ.get("ALPACA_API_KEY", "")
        or os.environ.get("APCA_API_KEY_ID", ""),
        alpaca_api_secret=os.environ.get("ALPACA_API_SECRET", "")
        or os.environ.get("APCA_API_SECRET_KEY", ""),
        fred_api_key=os.environ.get("FRED_API_KEY", ""),
    )


def load_config(path: Optional[Path | str] = None) -> GreeksConfig:
    """Load and validate greeks config. Only public entry for config I/O."""
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    # Allow override without callers reading env themselves for the path.
    env_path = os.environ.get("GREEKS_CONFIG")
    if path is None and env_path:
        cfg_path = Path(env_path)

    raw = _load_yaml(cfg_path)

    methodology_version = str(raw.get("methodology_version", "")).strip()
    if not methodology_version:
        raise ValueError("methodology_version is required")

    window_raw = _require_mapping(raw.get("window"), "window")
    start = _parse_date(str(window_raw["start"]))
    end = _parse_date(str(window_raw["end"]))
    excluded_raw = window_raw.get("excluded_dates", [])
    if not isinstance(excluded_raw, list):
        raise TypeError("window.excluded_dates must be a list")
    excluded = tuple(_parse_date(str(d)) for d in excluded_raw)
    window = ValidationWindow(start=start, end=end, excluded_dates=excluded)

    tickers_raw = raw.get("tickers", [])
    if not isinstance(tickers_raw, list) or not tickers_raw:
        raise ValueError("tickers must be a non-empty list")
    tickers = tuple(str(t).upper() for t in tickers_raw)

    t_floor_minutes = int(raw.get("t_floor_minutes", 15))
    join_staleness_s = int(raw.get("join_staleness_s", 300))
    if t_floor_minutes < 0:
        raise ValueError("t_floor_minutes must be >= 0")
    if join_staleness_s <= 0:
        raise ValueError("join_staleness_s must be > 0")

    expiry_time_et = str(raw.get("expiry_time_et", "16:00"))
    day_count = str(raw.get("day_count", "ACT/365"))
    if day_count != "ACT/365":
        raise ValueError(f"unsupported day_count: {day_count!r} (only ACT/365)")

    sofr_table = str(raw.get("sofr_table", "trading.sofr_daily"))
    div_raw = raw.get("dividends", {})
    if not isinstance(div_raw, Mapping):
        raise TypeError("dividends must be a mapping")
    dividends = _parse_dividends(div_raw)

    tables_raw = raw.get("tables", {})
    if not isinstance(tables_raw, Mapping):
        raise TypeError("tables must be a mapping")
    tables = _parse_tables(tables_raw, sofr_table=sofr_table)

    return GreeksConfig(
        methodology_version=methodology_version,
        window=window,
        tickers=tickers,
        t_floor_minutes=t_floor_minutes,
        join_staleness_s=join_staleness_s,
        expiry_time_et=expiry_time_et,
        day_count=day_count,
        sofr_table=sofr_table,
        dividends=dividends,
        tables=tables,
        clickhouse=_clickhouse_from_env(),
        api_keys=_api_keys_from_env(),
        config_path=cfg_path.resolve(),
    )


@lru_cache(maxsize=1)
def get_config() -> GreeksConfig:
    """Cached default config. Clear with ``get_config.cache_clear()`` in tests."""
    return load_config()
