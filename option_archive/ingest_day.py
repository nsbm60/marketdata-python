"""PR3 — per-day trade flat-file ingest.

The historical trade drain is one job per trading day: download that day's
whole-market OPRA trade file (<100 MB, gzipped CSV) from the vendor's S3, keep only
the contracts enumerated for that day, attach the as-of raw spot from one equity
tape per underlying, insert complete rows to ``trading.option_trade`` (quote columns
NULL — the enrichment pass fills them later), mark the day's jobs done, discard the
file. Per-contract REST (`greeks/pull/massive_trades`) is retained for PR5
incremental/new-name fills; it is not the bulk path.

Identity: the trade file has no ``sequence_number`` and can contain byte-identical
prints, so each print gets a stable within-(contract,day) ``ordinal`` in file order
(decision 7) — distinct same-nanosecond prints survive and re-ingest is idempotent.

No ``asyncio`` / ``threading`` (process fleet only). The network calls live in
module functions so tests inject fakes.
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Sequence

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from clickhouse_connect.driver.client import Client

from greeks.occ import parse_occ, strip_massive_prefix
from greeks.pull.alpaca_spot import EquityTradePrint, fetch_equity_trades
from greeks.pull.massive_trades import ns_to_utc
from option_archive.config import ArchiveConfig
from option_archive.domain import OsiSymbol, TaskFailure, WorkTask
from option_archive.queue import WorkQueue

log = logging.getLogger(__name__)

MASSIVE_S3_ENDPOINT = "https://files.massive.com"
FLATFILES_BUCKET = "flatfiles"
TRADES_KEY_TEMPLATE = "us_options_opra/trades_v1/{y:04d}/{m:02d}/{y:04d}-{m:02d}-{d:02d}.csv.gz"

# Trade flat-file columns (confirmed from a sample day).
# Massive documents an 8-column trade file, but some real files (e.g. 2024-05-23)
# ship only these 7, omitting participant_timestamp (discrepancy reported to
# Massive). We parse by header NAME and treat participant_timestamp as OPTIONAL —
# populated when present, NULL when not — so both layouts load.
_REQUIRED_COLUMNS = (
    "ticker", "conditions", "correction", "exchange", "price", "sip_timestamp", "size",
)
_PARTICIPANT_COLUMN = "participant_timestamp"


@dataclass(frozen=True)
class FlatTradePrint:
    """One row of the trade flat file, keyed for the archive by (symbol, sip, ordinal)."""

    symbol: OsiSymbol  # bare OSI
    ordinal: int  # stable within-(contract,day) position, file order
    sip_timestamp_ns: int
    participant_timestamp_ns: Optional[int]  # present only when the file carries the column
    price: float
    size: float
    exchange: Optional[int]
    conditions: tuple[int, ...]
    correction: Optional[int]

    @property
    def trade_ts(self) -> datetime:
        return ns_to_utc(self.sip_timestamp_ns)


@dataclass(frozen=True)
class OptionTradeRow:
    """A row for ``trading.option_trade`` (quote columns NULL at ingest)."""

    symbol: str
    underlying: str
    session_date: date
    trade_ts: datetime
    price: float
    size: float
    exchange: Optional[int]
    conditions: tuple[int, ...]
    sip_timestamp_ns: int
    ordinal: int
    sequence_number: Optional[int]  # None for flat-file rows
    participant_timestamp_ns: Optional[int]  # None when the file omits the column
    correction: Optional[int]
    spot_at_trade: Optional[float]
    spot_trade_ts: Optional[datetime]
    source: str = "massive"


TRANSPORT_FLATFILE = "flatfile"


@dataclass(frozen=True)
class LedgerRow:
    """One `trading.ingest_log` row — the durable record of a completed day."""

    session_date: date
    transport: str
    tasks_success: int
    tasks_no_trades: int
    tasks_failed: int
    rows_inserted: int
    bytes_downloaded: int
    wall_seconds: float
    vendor_volume_delta: Optional[int]  # None until acceptance check 2 runs
    started_at: datetime
    finished_at: datetime


@dataclass(frozen=True)
class DayReport:
    work_date: date
    contracts_claimed: int
    trades_inserted: int
    no_trade_contracts: int
    outcome: str  # "done" | "transport" | "vendor"


@dataclass(frozen=True)
class WorkerReport:
    days_done: int
    trades_inserted: int


# ---------------------------------------------------------------------------
# network seams (module-level so tests inject fakes)
# ---------------------------------------------------------------------------


def make_s3_client(
    access_key: str,
    secret_key: str,
    *,
    connect_timeout: float = 30.0,
    read_timeout: float = 600.0,
) -> object:
    """S3 client for the vendor's S3-compatible endpoint. Credentials come from env
    (never committed); constructed once and passed to the worker. read_timeout is
    generous: the botocore 60s default aborts a slow large download mid-stream on
    this throttled link — a client deadline shorter than the work needs."""
    if not access_key or not secret_key:
        raise ValueError("flat-file S3 access key and secret are required")
    session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    return session.client(
        "s3",
        endpoint_url=MASSIVE_S3_ENDPOINT,
        config=BotoConfig(
            signature_version="s3v4",
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        ),
    )


def download_trades_day(
    s3: object,
    work_date: date,
    *,
    bucket: str = FLATFILES_BUCKET,
    max_concurrency: int = 16,
    multipart_chunksize: int = 8 * 1024 * 1024,
    multipart_threshold: int = 8 * 1024 * 1024,
) -> bytes:
    """Fetch one day's gzipped trade file **into memory** via boto3 managed
    multipart transfer — the object is split into byte-range parts pulled over
    `max_concurrency` connections (~5x the single-stream rate on the throttled
    link). `download_fileobj` into a BytesIO keeps it in memory: still no disk temp,
    so nothing to clean up on any exit path (the buffer frees on scope exit)."""
    key = TRADES_KEY_TEMPLATE.format(y=work_date.year, m=work_date.month, d=work_date.day)
    cfg = TransferConfig(
        multipart_threshold=multipart_threshold,
        multipart_chunksize=multipart_chunksize,
        max_concurrency=max_concurrency,
    )
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf, Config=cfg)  # type: ignore[attr-defined]
    return buf.getvalue()


def equity_tape(alpaca: object, underlying: str, work_date: date) -> list[EquityTradePrint]:
    """RAW SIP equity tape for one underlying-day (fetched once, reused across that
    underlying's contracts)."""
    return fetch_equity_trades(alpaca, underlying, work_date)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse + join (pure)
# ---------------------------------------------------------------------------


def _opt_int(cell: str) -> Optional[int]:
    cell = cell.strip()
    return int(cell) if cell else None


def parse_trades(
    raw_gz: bytes, keep: frozenset[OsiSymbol]
) -> dict[OsiSymbol, list[FlatTradePrint]]:
    """Parse the gzipped trade CSV, keeping only ``keep`` contracts, assigning each
    kept print a stable ordinal in file order per contract."""
    text = gzip.decompress(raw_gz).decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None:
        raise ValueError("empty trade file (no header)")
    cols = [c.strip() for c in header]
    idx = {name: i for i, name in enumerate(cols)}
    missing = [c for c in _REQUIRED_COLUMNS if c not in idx]
    if missing:
        raise ValueError(f"trade file missing required columns {missing}: header={cols!r}")
    ncols = len(cols)
    p_idx = idx.get(_PARTICIPANT_COLUMN)  # None when the file omits participant_timestamp
    ti, ci, ri, ei, pi, si, zi = (
        idx["ticker"], idx["conditions"], idx["correction"], idx["exchange"],
        idx["price"], idx["sip_timestamp"], idx["size"],
    )
    out: dict[OsiSymbol, list[FlatTradePrint]] = defaultdict(list)
    for row in reader:
        if len(row) != ncols:
            raise ValueError(f"malformed trade row (got {len(row)} of {ncols}): {row!r}")
        sym = OsiSymbol(strip_massive_prefix(row[ti]))
        if sym not in keep:
            continue
        cond = _opt_int(row[ci])
        out[sym].append(
            FlatTradePrint(
                symbol=sym,
                ordinal=len(out[sym]),  # 0-based position within this contract, file order
                sip_timestamp_ns=int(row[si]),
                participant_timestamp_ns=(_opt_int(row[p_idx]) if p_idx is not None else None),
                price=float(row[pi]),
                size=float(row[zi]),
                exchange=_opt_int(row[ei]),
                conditions=(cond,) if cond is not None else (),
                correction=_opt_int(row[ri]),
            )
        )
    return dict(out)


def _attach_spot(
    trades_by_ts: Sequence[FlatTradePrint], tape: Sequence[EquityTradePrint]
) -> dict[tuple[str, int, int], EquityTradePrint]:
    """Merge-walk as-of join (last equity print at-or-before each trade), the same
    algorithm as ``attach_spot_to_option_trades``. ``trades_by_ts`` and ``tape`` must
    be sorted ascending by timestamp. Keyed by (symbol, sip_ns, ordinal)."""
    out: dict[tuple[str, int, int], EquityTradePrint] = {}
    j = 0
    last: Optional[EquityTradePrint] = None
    for t in trades_by_ts:
        while j < len(tape) and tape[j].trade_ts <= t.trade_ts:
            last = tape[j]
            j += 1
        if last is not None:
            out[(t.symbol, t.sip_timestamp_ns, t.ordinal)] = last
    return out


def _build_rows(
    by_symbol: dict[OsiSymbol, list[FlatTradePrint]],
    spot: dict[tuple[str, int, int], EquityTradePrint],
    work_date: date,
) -> list[OptionTradeRow]:
    rows: list[OptionTradeRow] = []
    for sym, prints in by_symbol.items():
        underlying = parse_occ(sym).root
        for p in prints:
            eq = spot.get((p.symbol, p.sip_timestamp_ns, p.ordinal))
            rows.append(
                OptionTradeRow(
                    symbol=sym,
                    underlying=underlying,
                    session_date=work_date,
                    trade_ts=p.trade_ts,
                    price=p.price,
                    size=p.size,
                    exchange=p.exchange,
                    conditions=p.conditions,
                    sip_timestamp_ns=p.sip_timestamp_ns,
                    ordinal=p.ordinal,
                    sequence_number=None,  # flat file supplies none
                    participant_timestamp_ns=p.participant_timestamp_ns,
                    correction=p.correction,
                    spot_at_trade=eq.price if eq is not None else None,
                    spot_trade_ts=eq.trade_ts if eq is not None else None,
                )
            )
    return rows


_INSERT_COLUMNS = (
    "symbol", "underlying", "session_date", "trade_ts", "price", "size", "exchange",
    "conditions", "sip_timestamp_ns", "ordinal", "sequence_number",
    "participant_timestamp_ns", "correction", "spot_at_trade", "spot_trade_ts",
    "quote_ts", "bid", "ask", "bid_size", "ask_size", "quote_lag_ms", "source",
)


def insert_option_trades(ch: Client, rows: Sequence[OptionTradeRow], *, table: str) -> int:
    """Batch-insert complete rows (quote columns NULL). ``ingested_at`` uses the CH
    DEFAULT now64(3), so it is not supplied here."""
    if not rows:
        return 0
    data = [
        [
            r.symbol, r.underlying, r.session_date, r.trade_ts, r.price, r.size,
            r.exchange, list(r.conditions), r.sip_timestamp_ns, r.ordinal,
            r.sequence_number, r.participant_timestamp_ns, r.correction,
            r.spot_at_trade, r.spot_trade_ts,
            None, None, None, None, None, None,  # quote_ts, bid, ask, bid_size, ask_size, quote_lag_ms
            r.source,
        ]
        for r in rows
    ]
    db, name = (table.split(".", 1) if "." in table else (None, table))
    ch.insert(name, data, column_names=list(_INSERT_COLUMNS), database=db)
    return len(rows)


_LEDGER_COLUMNS = (
    "session_date", "transport", "tasks_success", "tasks_no_trades", "tasks_failed",
    "rows_inserted", "bytes_downloaded", "wall_seconds", "vendor_volume_delta",
    "started_at", "finished_at",
)


def insert_ingest_log(ch: Client, row: LedgerRow, *, table: str) -> None:
    """Append one ledger row. Best-effort analytics written after mark_done — a
    failure here is logged, never rolled back onto the (already-done) tasks."""
    db, name = (table.split(".", 1) if "." in table else (None, table))
    data = [[
        row.session_date, row.transport, row.tasks_success, row.tasks_no_trades,
        row.tasks_failed, row.rows_inserted, row.bytes_downloaded, row.wall_seconds,
        row.vendor_volume_delta, row.started_at, row.finished_at,
    ]]
    ch.insert(name, data, column_names=list(_LEDGER_COLUMNS), database=db)


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------


def _claim_day(queue: WorkQueue, *, now: Optional[datetime]) -> list[WorkTask]:
    """Claim the oldest claimable day's tasks. Existing ORDER BY (work_date, contract)
    yields a day contiguously; a later-day claim is released back."""
    first = queue.claim_next(now=now)
    if first is None:
        return []
    day = first.work_date
    claimed = [first]
    while True:
        nxt = queue.claim_next(now=now)
        if nxt is None:
            break
        if nxt.work_date == day:
            claimed.append(nxt)
        else:
            queue.release_transport(nxt, "deferred: belongs to a later day-batch")
            break
    return claimed


def ingest_day(
    queue: WorkQueue,
    ch: Client,
    s3: object,
    alpaca: object,
    cfg: ArchiveConfig,
    *,
    now: Optional[datetime] = None,
) -> Optional[DayReport]:
    """Ingest one trading day's trades for the claimed contract-set. Returns None
    when the queue is drained."""
    claimed = _claim_day(queue, now=now)
    if not claimed:
        return None
    day = claimed[0].work_date
    keep = frozenset(t.contract for t in claimed)
    started_at = datetime.now(timezone.utc)
    t0 = time.monotonic()

    try:
        raw = download_trades_day(
            s3, day,
            max_concurrency=cfg.s3.max_concurrency,
            multipart_chunksize=cfg.s3.multipart_chunksize,
            multipart_threshold=cfg.s3.multipart_threshold,
        )
    except (ClientError, BotoCoreError, OSError) as e:
        for t in claimed:
            queue.release_transport(t, f"s3 download failed: {e}")
        log.warning("day %s: S3 download failed, released %d tasks: %s", day, len(claimed), e)
        return DayReport(day, len(claimed), 0, 0, "transport")

    try:
        by_symbol = parse_trades(raw, keep)
    except (ValueError, gzip.BadGzipFile) as e:
        for t in claimed:
            queue.fail_vendor(t, TaskFailure.VENDOR_ERROR, f"parse failed: {e}", now=now)
        log.error("day %s: trade file parse failed, failed %d tasks: %s", day, len(claimed), e)
        return DayReport(day, len(claimed), 0, 0, "vendor")

    try:
        spot: dict[tuple[str, int, int], EquityTradePrint] = {}
        by_underlying: dict[str, list[FlatTradePrint]] = defaultdict(list)
        for sym, prints in by_symbol.items():
            by_underlying[parse_occ(sym).root].extend(prints)
        for underlying, prints in by_underlying.items():
            tape = equity_tape(alpaca, underlying, day)
            spot.update(_attach_spot(sorted(prints, key=lambda p: p.trade_ts), tape))
    except (RuntimeError, OSError) as e:  # equity fetch = infra/transport
        for t in claimed:
            queue.release_transport(t, f"equity tape failed: {e}")
        log.warning("day %s: equity tape failed, released %d tasks: %s", day, len(claimed), e)
        return DayReport(day, len(claimed), 0, 0, "transport")

    rows = _build_rows(by_symbol, spot, day)

    try:
        inserted = insert_option_trades(ch, rows, table=cfg.tables.option_trade)
    except Exception as e:  # CH insert failure = infra; retry without penalty
        for t in claimed:
            queue.release_transport(t, f"clickhouse insert failed: {e}")
        log.warning("day %s: insert failed, released %d tasks: %s", day, len(claimed), e)
        return DayReport(day, len(claimed), 0, 0, "transport")

    # Commit succeeded — mark done (fenced). Contracts absent from the file are the
    # reasoned no-trades outcome: DONE with zero rows, no queue change.
    for t in claimed:
        queue.mark_done(t, now=now)
    no_trades = sum(1 for t in claimed if t.contract not in by_symbol)

    # Final act: the durable ledger row (what WAS DONE). Best-effort — the tasks
    # are already DONE, so a ledger failure is logged, not rolled back.
    ledger = LedgerRow(
        session_date=day,
        transport=TRANSPORT_FLATFILE,
        tasks_success=len(claimed) - no_trades,
        tasks_no_trades=no_trades,
        tasks_failed=0,
        rows_inserted=inserted,
        bytes_downloaded=len(raw),
        wall_seconds=time.monotonic() - t0,
        vendor_volume_delta=None,  # filled when acceptance check 2 runs
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
    )
    try:
        insert_ingest_log(ch, ledger, table=cfg.tables.ingest_log)
    except Exception as e:  # analytics only; do not un-mark done work
        log.warning("day %s: ingest_log write failed (tasks remain done): %s", day, e)

    log.info(
        "day %s: %d contracts, %d trades inserted, %d no-trades",
        day, len(claimed), inserted, no_trades,
    )
    return DayReport(day, len(claimed), inserted, no_trades, "done")


def run_worker(
    queue: WorkQueue,
    ch: Client,
    s3: object,
    alpaca: object,
    cfg: ArchiveConfig,
    *,
    poll: timedelta,
) -> WorkerReport:
    """Drain the queue day by day. Empty claim + pending → sleep(poll); empty +
    drained → return."""
    days = 0
    trades = 0
    while True:
        rep = ingest_day(queue, ch, s3, alpaca, cfg, now=datetime.now(timezone.utc))
        if rep is None:
            if queue.pending_exists():
                time.sleep(poll.total_seconds())
                continue
            break
        if rep.outcome == "done":
            days += 1
            trades += rep.trades_inserted
    return WorkerReport(days_done=days, trades_inserted=trades)
