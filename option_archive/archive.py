"""option_archive — the whole backfill in one program, run repeatedly.

Each run resolves ClickHouse via discovery, resumes from the ``trading.ingest_log``
ledger, and walks NYSE trading days forward to yesterday. For each day it:

  1. downloads the Massive whole-market OPRA trade file (managed multipart to a
     tempfile; retry with backoff; give up after a few tries and exit nonzero so
     the next run resumes here),
  2. parses it, keeping the watchlist names,
  3. enumerates each name's as-of chain (weekly sample) and writes it to
     ``option_contract_asof`` — the self-describing reference,
  4. keeps the in-band trades (ordinals already assigned in sip order), flags any
     in-band trade on a contract the listing omitted as an ENUMERATION_MISS
     (recorded + logged, not fatal — a vendor listing gap must not wedge the decade),
  5. inserts the trades into ``option_trade``,
  6. writes ONE ``ingest_log`` row as the completion marker — written last.

Crash safety is the ledger: a day is done iff its ledger row exists, and the row is
written after its trades, so a crash leaves a partial day with no ledger row; the
re-run redoes it and ReplacingMergeTree + per-contract ordinals collapse the
duplicates. No queue, no leases, no worker fleet — it runs at full speed when
started, on a nightly timer or manually.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

from boto3.s3.transfer import TransferConfig
from botocore.exceptions import BotoCoreError, ClientError
from clickhouse_connect.driver.client import Client

from greeks.domain import OptionRight
from greeks.occ import parse_occ
from greeks.pull.alpaca_spot import EquityTradePrint, make_stock_client
from greeks.pull.contracts import (
    ContractRef,
    dte_days,
    fetch_massive_contracts,
    filter_contracts,
    is_in_moneyness_band,
)
from ml.shared.clickhouse import get_ch_client  # canonical connector (discovery for all)
from option_archive.config import ArchiveConfig, get_config
from option_archive.domain import BandSpec, OsiSymbol
from option_archive.ingest_day import (
    FLATFILES_BUCKET,
    TRADES_KEY_TEMPLATE,
    TRANSPORT_FLATFILE,
    FlatTradePrint,
    LedgerRow,
    NbboQuote,
    OptionTradeRow,
    _attach_spot,
    _build_rows,
    attach_quotes,
    equity_tape,
    insert_ingest_log,
    insert_option_trades,
    make_s3_client,
    parse_trades,
)
from option_archive.quotes import fetch_option_quotes_day
from option_archive.reference import (
    _fetch_daily_raw_closes,
    _week_anchor,
    era_for,
    watchlist_underlyings,
)
from option_archive.retry import with_retry

log = logging.getLogger("option_archive")

_OVERLAP_DAYS = 3            # re-check window behind the ledger max (see _resume_start_date)
_DOWNLOAD_ATTEMPTS = 5
_BACKOFF_BASE_SEC = 5.0
_BACKOFF_MAX_SEC = 120.0
_IO_WORKERS = 8             # bounded pool for the per-name equity-tape / reference fetches


class DownloadExhausted(RuntimeError):
    """Raised after ``_DOWNLOAD_ATTEMPTS`` failed download attempts for one day."""


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


def _resume_start_date(ch: Client, cfg: ArchiveConfig) -> date:
    """Start ``_OVERLAP_DAYS`` behind the newest ledgered day (a small overlap
    re-check absorbs any day left half-written by a crash), or the config floor if
    the ledger is empty."""
    cnt, top = ch.query(
        f"SELECT count(), max(session_date) FROM {cfg.tables.ingest_log}"
    ).result_rows[0]
    if not cnt:
        return cfg.backfill_start_date
    top = top.date() if isinstance(top, datetime) else top
    return max(cfg.backfill_start_date, top - timedelta(days=_OVERLAP_DAYS))


def _done_sessions(ch: Client, cfg: ArchiveConfig) -> set[date]:
    """Session dates already in the ledger — skipped even inside the overlap window."""
    rows = ch.query(
        f"SELECT DISTINCT session_date FROM {cfg.tables.ingest_log}"
    ).result_rows
    return {(d.date() if isinstance(d, datetime) else d) for (d,) in rows}


# ---------------------------------------------------------------------------
# trading days + spot
# ---------------------------------------------------------------------------


def _load_spots(
    alpaca: object, underlyings: tuple[str, ...], start: date, end: date
) -> dict[str, dict[date, float]]:
    """Per-underlying RAW daily closes over the span (spot for moneyness)."""
    return {u: _fetch_daily_raw_closes(alpaca, u, start, end) for u in underlyings}  # type: ignore[arg-type]


def _trading_days(spots: dict[str, dict[date, float]], start: date, end: date) -> list[date]:
    """The NYSE session set = every date any watchlist underlying has a bar."""
    days: set[date] = set()
    for closes in spots.values():
        days.update(d for d in closes if start <= d <= end)
    return sorted(days)


# ---------------------------------------------------------------------------
# download (tempfile; cleaned up on every path)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _download_day(s3: object, cfg: ArchiveConfig, day: date) -> Iterator[Path]:
    """Download the day's trade file to a tempfile via managed multipart, retrying
    with backoff. The tempfile is removed on EVERY path: each failed attempt unlinks
    its partial file before retrying, the final give-up unlinks and raises
    ``DownloadExhausted``, and a successful ``yield`` unlinks in ``finally``."""
    key = TRADES_KEY_TEMPLATE.format(y=day.year, m=day.month, d=day.day)
    tcfg = TransferConfig(
        multipart_threshold=cfg.s3.multipart_threshold,
        multipart_chunksize=cfg.s3.multipart_chunksize,
        max_concurrency=cfg.s3.max_concurrency,
    )
    backoff = _BACKOFF_BASE_SEC
    path: Optional[Path] = None
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        fd, name = tempfile.mkstemp(prefix=f"optarch_{day.isoformat()}_", suffix=".csv.gz")
        os.close(fd)
        candidate = Path(name)
        try:
            with candidate.open("wb") as fh:
                s3.download_fileobj(FLATFILES_BUCKET, key, fh, Config=tcfg)  # type: ignore[attr-defined]
            path = candidate
            break
        except (ClientError, BotoCoreError, OSError) as e:
            candidate.unlink(missing_ok=True)
            if attempt == _DOWNLOAD_ATTEMPTS:
                raise DownloadExhausted(
                    f"{day}: download failed after {_DOWNLOAD_ATTEMPTS} attempts: {e}"
                ) from e
            log.warning("day %s: download attempt %d/%d failed: %s; retry in %.0fs",
                        day, attempt, _DOWNLOAD_ATTEMPTS, e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, _BACKOFF_MAX_SEC)
    assert path is not None  # loop either set path or raised
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# enumeration + as-of reference write
# ---------------------------------------------------------------------------

_ASOF_COLUMNS = (
    "underlying", "as_of_date", "option_symbol", "expiration_date", "strike_price",
    "contract_type", "exercise_style", "primary_exchange", "shares_per_contract",
)  # ccy, source, fetched_at use the table DEFAULTs (fetched_at = ReplacingMergeTree version)


def _right_code(right: OptionRight) -> str:
    return "C" if right == OptionRight.CALL else "P"


def _write_asof(ch: Client, underlying: str, as_of: date, listing: list[ContractRef], table: str) -> None:
    """Store the full as-of listing (before any band/standard filter) so the archive
    is self-describing. ReplacingMergeTree(fetched_at) collapses re-runs."""
    if not listing:
        return
    data = [
        [underlying, as_of, c.osi, c.expiry, c.strike, _right_code(c.right),
         c.exercise_style or "", c.primary_exchange or "", c.shares_per_contract]
        for c in listing
    ]
    db, name = (table.split(".", 1) if "." in table else (None, table))
    ch.insert(name, data, column_names=list(_ASOF_COLUMNS), database=db)


def _in_scope(osi: str, day: date, spot: float, band: BandSpec, excluded_wd: frozenset[int]) -> bool:
    """Whether a traded contract SHOULD have been enumerated (in-band by DTE +
    moneyness, allowed expiry weekday) — used to tell a real listing gap from a
    simply out-of-scope trade."""
    occ = parse_occ(osi)
    d = dte_days(day, occ.expiry)
    return (
        0 <= d <= band.max_dte_days
        and is_in_moneyness_band(occ.strike, spot, band.moneyness_band)
        and occ.expiry.weekday() not in excluded_wd
    )


# ---------------------------------------------------------------------------
# per-day processing
# ---------------------------------------------------------------------------


class _Plan(NamedTuple):
    """One watchlist underlying's per-day context (built single-threaded)."""

    underlying: str
    osimap: dict[OsiSymbol, list[FlatTradePrint]]
    spot: float
    band: BandSpec
    excluded_wd: frozenset[int]
    key: tuple[str, date, int]
    week: date


def _classify_day(
    by_symbol: dict[OsiSymbol, list[FlatTradePrint]],
    spots_for_day: dict[str, float],
    cfg: ArchiveConfig,
    api_key: str,
    ch: Client,
    day: date,
    cache: dict[tuple[str, date, int], list[ContractRef]],
) -> tuple[dict[OsiSymbol, list[FlatTradePrint]], int]:
    """Split each watchlist underlying's parsed prints into the set to insert
    (in-band, listed) and a count of ENUMERATION_MISS prints (in-band but the listing
    omitted the contract). Misses are logged + counted, never fatal.

    The weekly reference fetch is the slow part, so cache-miss fetches run on a
    bounded I/O pool. Workers ONLY call the vendor and return their listing — they
    touch no cache, no ClickHouse, no shared state. The as-of write and cache
    population happen after the join, single-threaded; per-day filtering is cheap
    and sequential."""
    by_underlying: dict[str, dict[OsiSymbol, list[FlatTradePrint]]] = defaultdict(dict)
    for osi, prints in by_symbol.items():
        by_underlying[parse_occ(osi).root][osi] = prints

    now = date.today()
    plans: list[_Plan] = []
    for underlying, osimap in by_underlying.items():
        spot = spots_for_day.get(underlying)
        if spot is None:
            log.debug("day %s: no daily bar for %s; skipping %d contracts",
                      day, underlying, len(osimap))
            continue
        band = cfg.band_for_underlying(underlying, era_for(day, now=now, cfg=cfg))
        week = _week_anchor(day)
        plans.append(_Plan(
            underlying, osimap, spot, band, cfg.excluded_expiry_weekdays(underlying),
            (underlying, week, band.max_dte_days), week,
        ))

    # Cache-miss fetches, concurrently. Fetch DTE is measured from the week anchor
    # but the per-day filter measures it per day (up to 6 days shorter), so pad the
    # fetch window by a week — otherwise a contract entering the band mid-week is
    # absent from `listed` and flagged a false miss. Per-day filtering stays unpadded.
    to_fetch = {p.key: p for p in plans if p.key not in cache}
    if to_fetch:
        def _fetch(p: _Plan) -> tuple[tuple[str, date, int], str, date, list[ContractRef]]:
            listing = with_retry(
                lambda: fetch_massive_contracts(
                    api_key, p.underlying, as_of=p.week, max_dte_days=p.band.max_dte_days + 7
                ),
                what=f"massive contracts {p.underlying}@{p.week}",
            )
            return p.key, p.underlying, p.week, listing

        with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
            for key, underlying, week, listing in pool.map(_fetch, list(to_fetch.values())):
                _write_asof(ch, underlying, week, listing, cfg.tables.option_contract_asof)
                cache[key] = listing

    keep: dict[OsiSymbol, list[FlatTradePrint]] = {}
    misses = 0
    for p in plans:
        listing = cache[p.key]
        result = filter_contracts(
            listing, as_of=day, spot=p.spot,
            max_dte_days=p.band.max_dte_days, moneyness_band=p.band.moneyness_band,
        )
        eligible = {c.osi for c in result.eligible if c.expiry.weekday() not in p.excluded_wd}
        listed = {c.osi for c in listing}
        missed: list[str] = []
        for osi, prints in p.osimap.items():
            if osi in eligible:
                keep[osi] = prints
            elif osi not in listed and _in_scope(osi, day, p.spot, p.band, p.excluded_wd):
                missed.append(osi)
                misses += len(prints)
        if missed:
            log.error(
                "day %s: %s ENUMERATION_MISS — %d in-band contracts traded but not "
                "listed as-of %s (e.g. %s)",
                day, p.underlying, len(missed), p.week, missed[:5],
            )
    return keep, misses


def _spots_for_day(
    alpaca: object, keep: dict[OsiSymbol, list[FlatTradePrint]], day: date
) -> dict[tuple[str, int, int], EquityTradePrint]:
    """As-of raw spot per print (one equity tape per underlying, pooled). Each worker
    fetches one underlying's tape and returns its OWN local as-of join — no shared
    state mutated inside a worker; the main thread merges the partials (keys unique
    per underlying, no collisions)."""
    by_underlying: dict[str, list[FlatTradePrint]] = defaultdict(list)
    for osi, prints in keep.items():
        by_underlying[parse_occ(osi).root].extend(prints)

    def _tape_join(item: tuple[str, list[FlatTradePrint]]) -> dict[tuple[str, int, int], EquityTradePrint]:
        underlying, prints = item
        tape = equity_tape(alpaca, underlying, day)
        return _attach_spot(sorted(prints, key=lambda p: p.trade_ts), tape)

    spot: dict[tuple[str, int, int], EquityTradePrint] = {}
    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
        for partial in pool.map(_tape_join, list(by_underlying.items())):
            spot.update(partial)
    return spot


def _quotes_for_day(
    cfg: ArchiveConfig, keep: dict[OsiSymbol, list[FlatTradePrint]], day: date
) -> dict[tuple[str, int, int], NbboQuote]:
    """As-of NBBO per print: pull each kept contract's day of quotes (REST /v3/quotes,
    pooled at cfg.quote_pool_size) and merge-walk to its prints. Only on/after
    quotes_available_from — before that no vendor quotes exist and every quote column
    stays NULL. Each worker pulls one contract and returns its own local join; the
    per-page 429/5xx/transport retry lives in the fetcher."""
    if day < cfg.quotes_available_from:
        return {}
    api_key = cfg.api_keys.massive_api_key

    def _one(item: tuple[OsiSymbol, list[FlatTradePrint]]) -> dict[tuple[str, int, int], NbboQuote]:
        osi, prints = item
        quotes = fetch_option_quotes_day(api_key, str(osi), day)
        return attach_quotes(prints, quotes)  # prints already sip-sorted (parse_trades)

    out: dict[tuple[str, int, int], NbboQuote] = {}
    with ThreadPoolExecutor(max_workers=cfg.quote_pool_size) as pool:
        for partial in pool.map(_one, list(keep.items())):
            out.update(partial)
    return out


def _process_day(
    ch: Client,
    alpaca: object,
    s3: object,
    cfg: ArchiveConfig,
    day: date,
    spots: dict[str, dict[date, float]],
    cache: dict[tuple[str, date, int], list[ContractRef]],
) -> tuple[LedgerRow, dict[str, float]]:
    """Download → parse → enumerate/store → keep → insert trades. Returns the ledger
    row (written last as the completion marker) and the per-phase seconds so the
    caller can log where the time actually went."""
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    watchlist = frozenset(spots.keys())
    with _download_day(s3, cfg, day) as path:
        raw = path.read_bytes()
    nbytes = len(raw)
    t_dl = time.monotonic()
    by_symbol = parse_trades(raw, watchlist)
    t_parse = time.monotonic()
    spots_for_day = {u: closes[day] for u, closes in spots.items() if day in closes}
    keep, misses = _classify_day(
        by_symbol, spots_for_day, cfg, cfg.api_keys.massive_api_key, ch, day, cache
    )
    t_enum = time.monotonic()  # enumerate covers the Massive reference fetch + as-of write
    quotes = _quotes_for_day(cfg, keep, day)
    t_quotes = time.monotonic()  # quotes covers the per-contract /v3/quotes pull (>= 2022-03-07)
    spot = _spots_for_day(alpaca, keep, day)
    t_tape = time.monotonic()  # tape covers the per-underlying equity-tape pull
    rows = _build_rows(keep, spot, quotes, day)
    inserted = insert_option_trades(ch, rows, table=cfg.tables.option_trade)
    t_ins = time.monotonic()
    phases = {
        "dl": t_dl - t0, "parse": t_parse - t_dl, "enum": t_enum - t_parse,
        "quotes": t_quotes - t_enum, "tape": t_tape - t_quotes, "insert": t_ins - t_tape,
    }
    ledger = LedgerRow(
        session_date=day,
        transport=TRANSPORT_FLATFILE,
        tasks_success=len(keep),
        tasks_no_trades=0,
        tasks_failed=0,
        rows_inserted=inserted,
        bytes_downloaded=nbytes,
        wall_seconds=t_ins - t0,
        vendor_volume_delta=None,
        enumeration_misses=misses,
        started_at=started,
        finished_at=datetime.now(timezone.utc),
    )
    return ledger, phases


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


def run(cfg: ArchiveConfig, ch: Client, alpaca: object, s3: object) -> int:
    """One pass: resume, walk trading days to yesterday, process each undone day.
    Returns a process exit code (0 caught up; 2 on download exhaustion)."""
    start = _resume_start_date(ch, cfg)
    yesterday = date.today() - timedelta(days=1)
    if start > yesterday:
        log.info("nothing to do: resume start %s is after yesterday %s", start, yesterday)
        return 0
    unders = watchlist_underlyings(ch, table=cfg.tables.watchlist)
    if not unders:
        log.error("watchlist %s is empty — nothing to archive", cfg.tables.watchlist)
        return 0
    log.info("resume from %s; watchlist=%d names; loading daily bars…", start, len(unders))
    spots = _load_spots(alpaca, unders, start, yesterday)
    done = _done_sessions(ch, cfg)
    days = [d for d in _trading_days(spots, start, yesterday) if d not in done]
    log.info("trading days to process: %d (through %s)", len(days), yesterday)

    cache: dict[tuple[str, date, int], list[ContractRef]] = {}
    total_rows = total_misses = 0
    try:
        for i, day in enumerate(days, 1):
            ledger, ph = _process_day(ch, alpaca, s3, cfg, day, spots, cache)
            insert_ingest_log(ch, ledger, table=cfg.tables.ingest_log)  # completion marker, LAST
            total_rows += ledger.rows_inserted
            total_misses += ledger.enumeration_misses
            log.info(
                "[%d/%d] %s: %d trades, %d misses, %.1fs "
                "(dl %.0f parse %.0f enum %.0f quotes %.0f tape %.0f ins %.0f)",
                i, len(days), day, ledger.rows_inserted, ledger.enumeration_misses,
                ledger.wall_seconds, ph["dl"], ph["parse"], ph["enum"], ph["quotes"],
                ph["tape"], ph["insert"],
            )
    except DownloadExhausted as e:
        log.error("%s — exiting; the next run resumes from this day", e)
        return 2
    log.info("caught up through %s: %d days, %d trades inserted, %d enumeration misses",
             yesterday, len(days), total_rows, total_misses)
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    # httpx logs each request URL at INFO, and vendor URLs carry the API key as a
    # query param — keep httpx at WARNING so the key never reaches the journal.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = get_config()
    ch = get_ch_client()
    alpaca = make_stock_client(cfg.api_keys.alpaca_api_key, cfg.api_keys.alpaca_api_secret)
    s3 = make_s3_client(
        os.environ.get("MASSIVE_S3_ACCESS_KEY", ""),
        os.environ.get("MASSIVE_S3_SECRET_KEY", ""),
        connect_timeout=cfg.s3.connect_timeout,
        read_timeout=cfg.s3.read_timeout,
    )
    sys.exit(run(cfg, ch, alpaca, s3))


if __name__ == "__main__":
    main()
