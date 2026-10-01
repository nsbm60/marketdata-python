"""Trade flat-file helpers: download client, parser, row build, and the two
ClickHouse inserts (`option_trade`, `ingest_log`).

These are the pure, reusable pieces the single-program `archive.py` orchestrates —
there is no queue, worker, or per-contract task here anymore. The network calls
live in module functions so tests inject fakes.

Identity: the trade file has no ``sequence_number`` and can contain byte-identical
prints, so each print gets a stable within-(contract,day) ``ordinal``. Prints are
sorted by ``sip_timestamp_ns`` before numbering (stable — equal timestamps keep
file order), because the vendor day file is not globally time-sorted.

No ``asyncio`` / ``threading`` (process fleet only).
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional, Sequence

import boto3
from botocore.config import Config as BotoConfig
from clickhouse_connect.driver.client import Client

from greeks.occ import parse_occ, strip_massive_prefix
from greeks.pull.alpaca_spot import EquityTradePrint, fetch_equity_trades
from greeks.pull.massive_trades import ns_to_utc
from option_archive.domain import OsiSymbol
from option_archive.retry import with_retry

log = logging.getLogger(__name__)

MASSIVE_S3_ENDPOINT = "https://files.massive.com"
FLATFILES_BUCKET = "flatfiles"
TRADES_KEY_TEMPLATE = "us_options_opra/trades_v1/{y:04d}/{m:02d}/{y:04d}-{m:02d}-{d:02d}.csv.gz"
TRANSPORT_FLATFILE = "flatfile"

# Trade flat-file columns (confirmed from a sample day). Massive documents an
# 8-column trade file, but some real files ship only these 7, omitting
# participant_timestamp (reported to Massive). We parse by header NAME and treat
# participant_timestamp as OPTIONAL — populated when present, NULL when not.
_REQUIRED_COLUMNS = (
    "ticker", "conditions", "correction", "exchange", "price", "sip_timestamp", "size",
)
_PARTICIPANT_COLUMN = "participant_timestamp"

# Cheap OSI root extractor for the per-row watchlist filter — avoids a full
# parse_occ on every row of a whole-market file (kept rows are parsed later).
_OSI_ROOT_RE = re.compile(r"^([A-Z]+)\d{6}[CP]\d{8}$")


@dataclass(frozen=True)
class FlatTradePrint:
    """One row of the trade flat file, keyed for the archive by (symbol, sip, ordinal)."""

    symbol: OsiSymbol  # bare OSI
    ordinal: int  # stable within-(contract,day) position, sip order (ties = file order)
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
    # as-of NBBO quote (last at-or-before the print's sip timestamp); NULL = vendor
    # had no quote, the only meaning. Filled inline by the quote phase.
    quote_ts: Optional[datetime]
    bid: Optional[float]
    ask: Optional[float]
    bid_size: Optional[int]
    ask_size: Optional[int]
    quote_lag_ms: Optional[int]
    source: str = "massive"


@dataclass(frozen=True)
class NbboQuote:
    """One NBBO quote for a contract (SIP clock). The fetcher lives in
    ``option_archive.quotes``; this is the record the quote phase joins to prints."""

    sip_timestamp_ns: int
    bid: Optional[float]
    ask: Optional[float]
    bid_size: Optional[int]
    ask_size: Optional[int]

    @property
    def quote_ts(self) -> datetime:
        return ns_to_utc(self.sip_timestamp_ns)


@dataclass(frozen=True)
class LedgerRow:
    """One `trading.ingest_log` row — the durable record of a completed day and the
    resume point. Written AFTER the day's trades, so its existence means the day is
    done."""

    session_date: date
    transport: str
    tasks_success: int
    tasks_no_trades: int
    tasks_failed: int
    rows_inserted: int
    bytes_downloaded: int
    wall_seconds: float
    vendor_volume_delta: Optional[int]  # None until acceptance check 2 runs
    enumeration_misses: int  # in-band trades on contracts the as-of listing omitted
    quote_contracts: int  # contracts the quote phase pulled (0 before 2022-03-07)
    quote_seconds: float  # quote-phase wall seconds
    started_at: datetime
    finished_at: datetime


# ---------------------------------------------------------------------------
# network seams (module-level so tests inject fakes)
# ---------------------------------------------------------------------------


def make_s3_client(
    access_key: str,
    secret_key: str,
    *,
    connect_timeout: float = 30.0,
    read_timeout: float = 600.0,
    max_attempts: Optional[int] = None,
) -> object:
    """S3 client for the vendor's S3-compatible endpoint. Credentials come from env
    (never committed); constructed once and passed in. read_timeout is generous: the
    botocore 60s default aborts a slow large download mid-stream on this throttled
    link — a client deadline shorter than the work needs. ``max_attempts`` overrides
    botocore's retry count (the probe passes 1 = no retry, for raw truth)."""
    if not access_key or not secret_key:
        raise ValueError("flat-file S3 access key and secret are required")
    kw: dict[str, Any] = dict(
        signature_version="s3v4",
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )
    if max_attempts is not None:
        kw["retries"] = {"max_attempts": max_attempts, "mode": "standard"}
    session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    return session.client("s3", endpoint_url=MASSIVE_S3_ENDPOINT, config=BotoConfig(**kw))


def equity_tape(alpaca: object, underlying: str, work_date: date) -> list[EquityTradePrint]:
    """RAW SIP equity tape for one underlying-day (fetched once, reused across that
    underlying's contracts). Retried on transient vendor errors."""
    return with_retry(
        lambda: fetch_equity_trades(alpaca, underlying, work_date),  # type: ignore[arg-type]
        what=f"alpaca equity tape {underlying}@{work_date}",
    )


# ---------------------------------------------------------------------------
# parse + join (pure)
# ---------------------------------------------------------------------------


def _opt_int(cell: str) -> Optional[int]:
    cell = cell.strip()
    return int(cell) if cell else None


def parse_trades(
    raw_gz: bytes, keep_underlyings: frozenset[str]
) -> dict[OsiSymbol, list[FlatTradePrint]]:
    """Parse the gzipped trade CSV, keeping only prints whose OSI root is in
    ``keep_underlyings`` (the watchlist). Per contract, prints are stable-sorted by
    ``sip_timestamp_ns`` (equal timestamps keep file order) and then numbered with a
    0-based ``ordinal`` — so re-ingest is idempotent and same-nanosecond prints
    survive as distinct rows."""
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
    # Collect raw fields per contract in file order; sort + number after.
    raw: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for row in reader:
        if len(row) != ncols:
            raise ValueError(f"malformed trade row (got {len(row)} of {ncols}): {row!r}")
        sym = strip_massive_prefix(row[ti])
        m = _OSI_ROOT_RE.match(sym)
        if m is None or m.group(1) not in keep_underlyings:
            continue
        cond = _opt_int(row[ci])
        raw[sym].append((
            int(row[si]),                                            # sip_timestamp_ns
            _opt_int(row[p_idx]) if p_idx is not None else None,     # participant_timestamp_ns
            float(row[pi]),                                          # price
            float(row[zi]),                                          # size
            _opt_int(row[ei]),                                       # exchange
            (cond,) if cond is not None else (),                     # conditions
            _opt_int(row[ri]),                                       # correction
        ))
    out: dict[OsiSymbol, list[FlatTradePrint]] = {}
    for sym, recs in raw.items():
        recs.sort(key=lambda r: r[0])  # stable: equal sip keeps file order
        osym = OsiSymbol(sym)
        out[osym] = [
            FlatTradePrint(
                symbol=osym, ordinal=i, sip_timestamp_ns=r[0],
                participant_timestamp_ns=r[1], price=r[2], size=r[3],
                exchange=r[4], conditions=r[5], correction=r[6],
            )
            for i, r in enumerate(recs)
        ]
    return out


def _attach_spot(
    trades_by_ts: Sequence[FlatTradePrint], tape: Sequence[EquityTradePrint]
) -> dict[tuple[str, int, int], EquityTradePrint]:
    """Merge-walk as-of join (last equity print at-or-before each trade).
    ``trades_by_ts`` and ``tape`` must be sorted ascending by timestamp. Keyed by
    (symbol, sip_ns, ordinal)."""
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


def attach_quotes(
    prints: Sequence[FlatTradePrint], quotes: Sequence[NbboQuote]
) -> dict[tuple[str, int, int], NbboQuote]:
    """Merge-walk as-of join: last quote at-or-before each print's sip timestamp.
    Both must be sorted ascending by ``sip_timestamp_ns``. Keyed by
    (symbol, sip_ns, ordinal)."""
    out: dict[tuple[str, int, int], NbboQuote] = {}
    j = 0
    last: Optional[NbboQuote] = None
    for p in prints:
        while j < len(quotes) and quotes[j].sip_timestamp_ns <= p.sip_timestamp_ns:
            last = quotes[j]
            j += 1
        if last is not None:
            out[(p.symbol, p.sip_timestamp_ns, p.ordinal)] = last
    return out


def _build_rows(
    by_symbol: dict[OsiSymbol, list[FlatTradePrint]],
    spot: dict[tuple[str, int, int], EquityTradePrint],
    quotes: dict[tuple[str, int, int], NbboQuote],
    work_date: date,
) -> list[OptionTradeRow]:
    rows: list[OptionTradeRow] = []
    for sym, prints in by_symbol.items():
        underlying = parse_occ(sym).root
        for p in prints:
            key = (p.symbol, p.sip_timestamp_ns, p.ordinal)
            eq = spot.get(key)
            q = quotes.get(key)
            # quote is at-or-before the print, so lag >= 0; NULL when there was no quote.
            lag = None if q is None else (p.sip_timestamp_ns - q.sip_timestamp_ns) // 1_000_000
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
                    quote_ts=q.quote_ts if q is not None else None,
                    bid=q.bid if q is not None else None,
                    ask=q.ask if q is not None else None,
                    bid_size=q.bid_size if q is not None else None,
                    ask_size=q.ask_size if q is not None else None,
                    quote_lag_ms=lag,
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
    """Batch-insert complete rows — quote columns filled by the quote phase, NULL
    only where the vendor had no quote. ``ingested_at`` uses the CH DEFAULT now64(3)."""
    if not rows:
        return 0
    data = [
        [
            r.symbol, r.underlying, r.session_date, r.trade_ts, r.price, r.size,
            r.exchange, list(r.conditions), r.sip_timestamp_ns, r.ordinal,
            r.sequence_number, r.participant_timestamp_ns, r.correction,
            r.spot_at_trade, r.spot_trade_ts,
            r.quote_ts, r.bid, r.ask, r.bid_size, r.ask_size, r.quote_lag_ms,
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
    "enumeration_misses", "quote_contracts", "quote_seconds", "started_at", "finished_at",
)


def insert_ingest_log(ch: Client, row: LedgerRow, *, table: str) -> None:
    """Append one ledger row — the day's completion marker, written last."""
    db, name = (table.split(".", 1) if "." in table else (None, table))
    data = [[
        row.session_date, row.transport, row.tasks_success, row.tasks_no_trades,
        row.tasks_failed, row.rows_inserted, row.bytes_downloaded, row.wall_seconds,
        row.vendor_volume_delta, row.enumeration_misses, row.quote_contracts,
        row.quote_seconds, row.started_at, row.finished_at,
    ]]
    ch.insert(name, data, column_names=list(_LEDGER_COLUMNS), database=db)
