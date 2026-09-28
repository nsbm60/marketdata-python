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
class S3Transfer:
    """Flat-file S3 client + managed-transfer tuning. The vendor throttles a single
    connection (~0.9 MB/s measured); managed multipart across `max_concurrency`
    connections reached ~4.7 MB/s. `read_timeout` is generous because the 60s default
    aborts a slow large download mid-stream."""

    connect_timeout: float  # seconds
    read_timeout: float  # seconds
    max_concurrency: int
    multipart_chunksize: int  # bytes
    multipart_threshold: int  # bytes

    def __post_init__(self) -> None:
        if self.connect_timeout <= 0 or self.read_timeout <= 0:
            raise ValueError("s3 timeouts must be > 0")
        if self.max_concurrency < 1:
            raise ValueError("s3.max_concurrency must be >= 1")
        if self.multipart_chunksize <= 0 or self.multipart_threshold <= 0:
            raise ValueError("s3 multipart sizes must be > 0")


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
    band_overrides: Mapping[str, BandSpec]  # per-underlying, applies in BOTH eras
    expiry_weekday_exclude: Mapping[str, frozenset[int]]  # per-underlying, Python weekday ints
    quotes_band: BandSpec
    backfill_start_date: date       # oldest date the queue enumerates (trades exist to 2014; we start 2022)
    quotes_available_from: date     # quotes only pulled on/after this (quote history floor)
    excluded_dates: tuple[date, ...]
    queue: QueuePolicy
    s3: S3Transfer
    roll_off: RollOffAlerting
    schedule: Schedule
    queue_db_path: Path
    tables: ArchiveTableNames
    api_keys: ApiKeys
    config_path: Path

    def band_for(self, era: Era) -> BandSpec:
        return self.bands[era]

    def band_for_underlying(self, underlying: str, era: Era) -> BandSpec:
        """Per-underlying override (applies in BOTH eras — no wide perishable band)
        if configured, else the per-era band. SPY/QQQ use a tight ±10%/90 override:
        their ATM/term-structure core is wanted, the deep wings are a non-goal."""
        override = self.band_overrides.get(underlying.upper())
        return override if override is not None else self.band_for(era)

    def excluded_expiry_weekdays(self, underlying: str) -> frozenset[int]:
        """Python weekday ints (Mon=0 … Sun=6) whose expirations are dropped for this
        underlying. SPY/QQQ exclude Tue/Thu: their count is dominated by short-dated
        expirations, and the Tue/Thu dailies (dense 2026-forward) are the reduction
        lever — moneyness is not."""
        return self.expiry_weekday_exclude.get(underlying.upper(), frozenset())


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


def _parse_band_overrides(raw: Any) -> dict[str, BandSpec]:
    if raw is None:
        return {}
    m = _require_mapping(raw, "band_overrides")
    out: dict[str, BandSpec] = {}
    for underlying, spec in m.items():
        s = _require_mapping(spec, f"band_overrides.{underlying}")
        out[str(underlying).upper()] = BandSpec(
            moneyness_band=float(s["moneyness_pct"]) / 100.0,
            max_dte_days=int(s["dte_max"]),
        )
    return out


_WEEKDAY = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


def _parse_expiry_weekday_exclude(raw: Any) -> dict[str, frozenset[int]]:
    if raw is None:
        return {}
    m = _require_mapping(raw, "expiry_weekday_exclude")
    out: dict[str, frozenset[int]] = {}
    for underlying, days in m.items():
        if not isinstance(days, list):
            raise TypeError(f"expiry_weekday_exclude.{underlying} must be a list")
        wds: set[int] = set()
        for d in days:
            key = str(d).strip().upper()[:3]
            if key not in _WEEKDAY:
                raise ValueError(f"expiry_weekday_exclude.{underlying}: bad weekday {d!r}")
            wds.add(_WEEKDAY[key])
        out[str(underlying).upper()] = frozenset(wds)
    return out


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


def _parse_s3(raw: Any) -> S3Transfer:
    m = _require_mapping(raw, "s3")
    return S3Transfer(
        connect_timeout=float(m.get("connect_timeout_seconds", 30)),
        read_timeout=float(m.get("read_timeout_seconds", 600)),
        max_concurrency=int(m.get("max_concurrency", 16)),
        multipart_chunksize=int(m.get("multipart_chunksize_mb", 8)) * 1024 * 1024,
        multipart_threshold=int(m.get("multipart_threshold_mb", 8)) * 1024 * 1024,
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
        band_overrides=_parse_band_overrides(raw.get("band_overrides")),
        expiry_weekday_exclude=_parse_expiry_weekday_exclude(raw.get("expiry_weekday_exclude")),
        quotes_band=_parse_band(raw.get("quotes_band"), "quotes_band"),
        backfill_start_date=backfill_start_date,
        quotes_available_from=quotes_available_from,
        excluded_dates=excluded,
        queue=_parse_queue(raw.get("queue")),
        s3=_parse_s3(raw.get("s3")),
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
