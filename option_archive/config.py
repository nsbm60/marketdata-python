"""Sole config-parsing chokepoint for the option_archive package.

All configuration loads through :func:`load_config`. Other modules must not read
env vars or YAML for archive settings except via the returned
:class:`ArchiveConfig`. Mirrors ``greeks/config.py``.

ClickHouse is *not* configured here: the archive connects via
``greeks.ch.get_ch_client`` (service discovery), so this file carries only
table-name pointers and the vendor API keys the pull needs. Each package owns its
own config chokepoint — the small env read below is intentional, not shared state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from greeks.config import ApiKeys  # reuse the frozen credential type
from option_archive.domain import BandSpec, Era, ScheduleWindowKind

# marketdata-python/ (repo root); config default sits at config/option_archive.yaml
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "option_archive.yaml"


@dataclass(frozen=True)
class UniverseParams:
    """Screened-universe parameters (the flat-file screen itself is deferred)."""

    top_n: int
    ranking_years: tuple[int, ...]
    index_products: tuple[str, ...]  # excluded from the union (SPY, QQQ, …)
    watchlist_always_include: bool

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError(f"universe.top_n must be > 0, got {self.top_n!r}")
        if not self.ranking_years:
            raise ValueError("universe.ranking_years must be non-empty")


@dataclass(frozen=True)
class ScheduleWindow:
    """One pacing regime that begins at ``start_et`` (America/New_York) and runs
    until the next window starts (see :class:`Schedule`).

    There is deliberately **no** end time: a window is defined by where it begins,
    and the day is a sequence of boundaries. That makes overlap and gaps
    unrepresentable — the thing two start/end pairs could never guarantee.
    ``start_et`` is a real ``time``, parsed at load, not an "HH:MM" string.
    """

    kind: ScheduleWindowKind
    start_et: time
    requests_per_sec: float
    worker_count: int

    def __post_init__(self) -> None:
        if self.requests_per_sec <= 0:
            raise ValueError(f"{self.kind.value}: requests_per_sec must be > 0")
        if self.worker_count <= 0:
            raise ValueError(f"{self.kind.value}: worker_count must be > 0")


@dataclass(frozen=True)
class Schedule:
    """The pacing day as clock boundaries. Each window runs from its ``start_et``
    to the next window's, wrapping past midnight, so exactly one window is active
    at any instant — no overlap, no gap, nothing to keep consistent by hand.
    """

    windows: tuple[ScheduleWindow, ...]

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("schedule must have at least one window")
        starts = [w.start_et for w in self.windows]
        if len(set(starts)) != len(starts):
            raise ValueError("schedule windows must have distinct start_et")
        if starts != sorted(starts):
            raise ValueError("schedule windows must be ordered by start_et")

    def active_at(self, t: time) -> ScheduleWindow:
        """The window in force at wall-clock ``t``: the latest one that has begun.
        Before the first boundary, the last window is active — it began the prior
        day and wraps midnight."""
        active = self.windows[-1]
        for w in self.windows:
            if w.start_et <= t:
                active = w
            else:
                break
        return active


@dataclass(frozen=True)
class QueuePolicy:
    """Work-queue tuning: claim lease, vendor-failure retry cap, backoff base."""

    lease: timedelta
    max_attempts: int
    backoff_base: timedelta

    def __post_init__(self) -> None:
        if self.lease <= timedelta(0):
            raise ValueError("queue.lease_seconds must be > 0")
        if self.max_attempts < 1:
            raise ValueError("queue.max_attempts must be >= 1")
        if self.backoff_base <= timedelta(0):
            raise ValueError("queue.backoff_base_seconds must be > 0")


@dataclass(frozen=True)
class RollOffAlerting:
    """Worst-case retention assumption and the alert margin against roll-off."""

    assumed_retention_years: int
    alert_margin_days: int

    def __post_init__(self) -> None:
        if self.assumed_retention_years <= 0:
            raise ValueError("roll_off.assumed_retention_years must be > 0")
        if self.alert_margin_days <= 0:
            raise ValueError("roll_off.alert_margin_days must be > 0")


@dataclass(frozen=True)
class ArchiveTableNames:
    """Fully-qualified CH table names. DDL lives in the Scala schema."""

    option_trade: str  # trades carry the as-of quote on the same row
    split: str
    universe_ranking: str
    ingest_log: str  # append-only ledger of completed ingest days
    dividend: str  # existing table, reused
    option_contract: str  # existing table, cross-check
    watchlist: str  # existing table, seed source (watchlist-first)


@dataclass(frozen=True)
class ArchiveConfig:
    """Immutable application config for the option archive."""

    universe: UniverseParams
    bands: Mapping[Era, BandSpec]
    quotes_band: BandSpec
    backfill_start_date: date       # oldest date the queue enumerates (trades exist to 2014; we start 2022)
    quotes_available_from: date     # quotes only pulled on/after this (quote history floor)
    excluded_dates: tuple[date, ...]
    queue: QueuePolicy
    roll_off: RollOffAlerting
    schedule: Schedule
    queue_db_path: Path
    tables: ArchiveTableNames
    api_keys: ApiKeys
    config_path: Path

    def band_for(self, era: Era) -> BandSpec:
        return self.bands[era]


def _require_mapping(raw: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"config key '{key}' must be a mapping")
    return raw


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"option_archive config not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise TypeError(f"option_archive config root must be a mapping: {path}")
    return data


def _parse_band(raw: Any, key: str) -> BandSpec:
    m = _require_mapping(raw, key)
    if "moneyness_band" not in m or "max_dte_days" not in m:
        raise ValueError(f"{key} requires moneyness_band and max_dte_days")
    return BandSpec(
        moneyness_band=float(m["moneyness_band"]),
        max_dte_days=int(m["max_dte_days"]),
    )


def _parse_bands(raw: Any) -> dict[Era, BandSpec]:
    m = _require_mapping(raw, "bands")
    bands: dict[Era, BandSpec] = {}
    for era in (Era.PERISHABLE, Era.ROUTINE):
        if era.value not in m:
            raise ValueError(f"bands.{era.value} is required")
        bands[era] = _parse_band(m[era.value], f"bands.{era.value}")
    return bands


def _parse_universe(raw: Any) -> UniverseParams:
    m = _require_mapping(raw, "universe")
    return UniverseParams(
        top_n=int(m.get("top_n", 100)),
        ranking_years=tuple(int(y) for y in m.get("ranking_years", [])),
        index_products=tuple(str(s).upper() for s in m.get("index_products", [])),
        watchlist_always_include=bool(m.get("watchlist_always_include", True)),
    )


def _parse_schedule(raw: Any) -> Schedule:
    if not isinstance(raw, list) or not raw:
        raise ValueError("schedule must be a non-empty list")
    out: list[ScheduleWindow] = []
    for item in raw:
        m = _require_mapping(item, "schedule[]")
        # ScheduleWindowKind(...) and time.fromisoformat(...) both reject bad input
        # here, at the boundary — never carried forward as an unvalidated string.
        out.append(
            ScheduleWindow(
                kind=ScheduleWindowKind(str(m["kind"])),
                start_et=time.fromisoformat(str(m["start_et"])),
                requests_per_sec=float(m["requests_per_sec"]),
                worker_count=int(m["worker_count"]),
            )
        )
    # sort by boundary so config order is irrelevant; Schedule validates the rest.
    out.sort(key=lambda w: w.start_et)
    return Schedule(windows=tuple(out))


def _parse_roll_off(raw: Any) -> RollOffAlerting:
    m = _require_mapping(raw, "roll_off")
    return RollOffAlerting(
        assumed_retention_years=int(m.get("assumed_retention_years", 5)),
        alert_margin_days=int(m.get("alert_margin_days", 60)),
    )


def _parse_queue(raw: Any) -> QueuePolicy:
    m = _require_mapping(raw, "queue")
    return QueuePolicy(
        lease=timedelta(seconds=int(m["lease_seconds"])),
        max_attempts=int(m["max_attempts"]),
        backoff_base=timedelta(seconds=int(m["backoff_base_seconds"])),
    )


def _parse_tables(raw: Any) -> ArchiveTableNames:
    m = _require_mapping(raw, "tables")
    return ArchiveTableNames(
        option_trade=str(m.get("option_trade", "trading.option_trade")),
        split=str(m.get("split", "trading.split")),
        universe_ranking=str(
            m.get("universe_ranking", "trading.option_universe_ranking")
        ),
        ingest_log=str(m.get("ingest_log", "trading.ingest_log")),
        dividend=str(m.get("dividend", "trading.dividend")),
        option_contract=str(m.get("option_contract", "trading.option_contract")),
        watchlist=str(m.get("watchlist", "trading.watchlist")),
    )


def _parse_queue_db_path(raw: Any) -> Path:
    """Resolve and enforce the ruling: the queue DB lives OUTSIDE the repo."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("queue_db_path is required")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        raise ValueError(
            f"queue_db_path must be absolute and outside the repo, got {raw!r}"
        )
    try:
        p.resolve().relative_to(_REPO_ROOT)
        inside_repo = True
    except ValueError:
        inside_repo = False
    if inside_repo:
        raise ValueError(
            f"queue_db_path must be OUTSIDE the repo ({_REPO_ROOT}); got {p}. "
            "The queue DB is never committed (Resolved decision 1)."
        )
    return p


def _api_keys_from_env() -> ApiKeys:
    # Same env vars as greeks.config; each package reads its own credentials.
    return ApiKeys(
        massive_api_key=os.environ.get("MASSIVE_API_KEY", "")
        or os.environ.get("POLYGON_API_KEY", ""),
        alpaca_api_key=os.environ.get("ALPACA_API_KEY", "")
        or os.environ.get("APCA_API_KEY_ID", ""),
        alpaca_api_secret=os.environ.get("ALPACA_API_SECRET", "")
        or os.environ.get("APCA_API_SECRET_KEY", ""),
        fred_api_key=os.environ.get("FRED_API_KEY", ""),
    )


def load_config(path: Optional[Path | str] = None) -> ArchiveConfig:
    """Load and validate archive config. Only public entry for config I/O."""
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    env_path = os.environ.get("OPTION_ARCHIVE_CONFIG")
    if path is None and env_path:
        cfg_path = Path(env_path)

    raw = _load_yaml(cfg_path)

    excluded_raw = raw.get("excluded_dates", [])
    if not isinstance(excluded_raw, list):
        raise TypeError("excluded_dates must be a list")
    excluded = tuple(date.fromisoformat(str(d)) for d in excluded_raw)

    if "backfill_start_date" not in raw:
        raise ValueError("backfill_start_date is required")
    if "quotes_available_from" not in raw:
        raise ValueError("quotes_available_from is required")
    backfill_start_date = date.fromisoformat(str(raw["backfill_start_date"]))
    quotes_available_from = date.fromisoformat(str(raw["quotes_available_from"]))

    return ArchiveConfig(
        universe=_parse_universe(raw.get("universe")),
        bands=_parse_bands(raw.get("bands")),
        quotes_band=_parse_band(raw.get("quotes_band"), "quotes_band"),
        backfill_start_date=backfill_start_date,
        quotes_available_from=quotes_available_from,
        excluded_dates=excluded,
        queue=_parse_queue(raw.get("queue")),
        roll_off=_parse_roll_off(raw.get("roll_off")),
        schedule=_parse_schedule(raw.get("schedule")),
        queue_db_path=_parse_queue_db_path(raw.get("queue_db_path")),
        tables=_parse_tables(raw.get("tables")),
        api_keys=_api_keys_from_env(),
        config_path=cfg_path.resolve(),
    )


@lru_cache(maxsize=1)
def get_config() -> ArchiveConfig:
    """Cached default config. Clear with ``get_config.cache_clear()`` in tests."""
    return load_config()
