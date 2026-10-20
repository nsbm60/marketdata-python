"""ClickHouse access for the greeks package (sync only).

Uses the same discovery path as ``ml.shared.clickhouse`` so runtime
location of CH is not hardcoded. Callers pass a client from
:func:`get_ch_client` — no module-level connections.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from greeks.domain import Dividend
from greeks.harness.rows import ResidualRow, ValidationRow, VendorSnapshot
from greeks.harness.store import residual_to_ch_dict, validation_to_ch_dict


def get_ch_client() -> Client:
    """Resolve ClickHouse via service discovery (same as ml.shared)."""
    from discovery import EndpointService, ServiceLocator

    ep = ServiceLocator.endpoint(
        EndpointService.CLICKHOUSE,
        timeout_sec=60,
    )
    return clickhouse_connect.get_client(
        host=ep.host,
        port=ep.port,
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
        database=os.environ.get("CLICKHOUSE_DATABASE", "trading"),
    )


def _aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _opt_float(v: object) -> Optional[float]:
    if v is None:
        return None
    return float(v)  # type: ignore[arg-type]


def load_sofr_series_from_ch(
    client: Client,
    *,
    table: str = "trading.sofr_daily",
    start: Optional[date] = None,
    end: Optional[date] = None,
    source: str = "nyfed",
) -> dict[date, float]:
    """Read SOFR rates from ClickHouse into ``date -> decimal rate``.

    **No silent fallback.** Uses ``FINAL`` (ReplacingMergeTree) and requires an
    explicit ``source`` (default ``nyfed``: the Scala BatchService sofr job, the table's only
    writer). Fixture and old ``fred`` rows are never mixed in.
    Raises ``ValueError`` if the query returns no rows.
    """
    if not source or not str(source).strip():
        raise ValueError("source is required (e.g. 'nyfed'); refusing unscoped load")
    src = str(source).strip()
    # FINAL so ReplacingMergeTree keeps latest version per date; source filter
    # so fixture/fred never collapse into an ambiguous dict overwrite.
    sql = f"SELECT date, rate, source FROM {table} FINAL WHERE source = {{src:String}}"
    params: dict[str, Any] = {"src": src}
    if start is not None:
        sql += " AND date >= {start:Date}"
        params["start"] = start
    if end is not None:
        sql += " AND date <= {end:Date}"
        params["end"] = end
    sql += " ORDER BY date"
    result = client.query(sql, parameters=params)
    out: dict[date, float] = {}
    for row in result.result_rows:
        d = row[0]
        if isinstance(d, datetime):
            d = d.date()
        row_source = str(row[2])
        if row_source != src:
            raise ValueError(
                f"SOFR row date={d} has source={row_source!r}, expected {src!r}"
            )
        rate = float(row[1])
        if d in out:
            raise ValueError(
                f"duplicate SOFR date {d} after FINAL for source={src!r}"
            )
        out[d] = rate
    if not out:
        raise ValueError(
            f"no SOFR rows in {table} for source={src!r}"
            + (f" start={start}" if start is not None else "")
            + (f" end={end}" if end is not None else "")
            + " — is the BatchService sofr job running? Will not fall back to fixture"
        )
    return out


def load_upcoming_dividends(
    client: Client,
    *,
    after: date,
    table: str = "trading.dividend",
) -> dict[str, tuple[Dividend, ...]]:
    """Cash dividends with ex-date strictly after ``after``, by underlying symbol.

    ``trading.dividend`` is maintained by the Scala DividendJob, which replaces a
    symbol's not-yet-ex rows on every run, so upcoming rows are not duplicated.
    A symbol with no upcoming dividend is simply absent from the result.
    """
    sql = (
        f"SELECT symbol, ex_date, amount FROM {table} "
        "WHERE ex_date > {after:Date} ORDER BY symbol, ex_date"
    )
    result = client.query(sql, parameters={"after": after})
    out: dict[str, list[Dividend]] = {}
    for row in result.result_rows:
        sym = str(row[0]).upper()
        ex = row[1]
        if isinstance(ex, datetime):
            ex = ex.date()
        amount = float(row[2])
        if not (amount >= 0.0):
            raise ValueError(f"{table}: {sym} ex_date={ex} has amount {amount!r}")
        out.setdefault(sym, []).append(Dividend(underlying=sym, amount=amount, ex_date=ex))
    return {k: tuple(v) for k, v in out.items()}


def fetch_vendor_snapshots(
    client: Client,
    *,
    underlying: str,
    session_date: date,
    symbols: Sequence[str],
    table: str = "trading.option_snapshot",
) -> list[VendorSnapshot]:
    """Pull vendor greeks snapshots for one underlying-day and symbol set.

    Capture window is the ET calendar day in UTC bounds
    ``[session_date 00:00 ET, next day)`` approximated as UTC day bounds
    spanning premarket through after-hours by using
    ``session_date`` 00:00 UTC through ``session_date+1`` 23:59 UTC is wrong
    for US session — use a wide UTC window covering the full US equity day:
    ``session_date`` 00:00 America/New_York → next day 00:00 ET, queried in UTC.
    """
    from zoneinfo import ZoneInfo

    if not symbols:
        return []
    et = ZoneInfo("America/New_York")
    start_local = datetime(
        session_date.year, session_date.month, session_date.day, tzinfo=et
    )
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)
    und = underlying.upper()
    syms = sorted({s.upper() for s in symbols})

    # clickhouse-connect named params for IN list
    sql = f"""
    SELECT
        symbol,
        underlying,
        timestamp,
        quote_timestamp,
        bid,
        ask,
        iv,
        delta,
        gamma,
        vega,
        theta,
        rho
    FROM {table}
    WHERE underlying = {{und:String}}
      AND timestamp >= {{start:DateTime64(3)}}
      AND timestamp < {{end:DateTime64(3)}}
      AND symbol IN {{syms:Array(String)}}
    ORDER BY symbol, timestamp
    """
    result = client.query(
        sql,
        parameters={
            "und": und,
            "start": start_utc,
            "end": end_utc,
            "syms": syms,
        },
    )
    out: list[VendorSnapshot] = []
    for row in result.result_rows:
        ts = _aware(row[2])
        qts = row[3]
        out.append(
            VendorSnapshot(
                symbol=str(row[0]),
                underlying=str(row[1]),
                timestamp=ts,
                quote_timestamp=_aware(qts) if qts is not None else None,
                bid=_opt_float(row[4]),
                ask=_opt_float(row[5]),
                iv=_opt_float(row[6]),
                delta=_opt_float(row[7]),
                gamma=_opt_float(row[8]),
                vega=_opt_float(row[9]),
                theta=_opt_float(row[10]),
                rho=_opt_float(row[11]),
            )
        )
    return out


def insert_validation_rows(
    client: Client,
    rows: Sequence[ValidationRow],
    *,
    table: str = "trading.greeks_validation",
) -> int:
    if not rows:
        return 0
    dicts = [validation_to_ch_dict(r) for r in rows]
    cols = list(dicts[0].keys())
    data = [[d[c] for c in cols] for d in dicts]
    db, name = _split_table(table)
    client.insert(name, data, column_names=cols, database=db)
    return len(rows)


def insert_residual_rows(
    client: Client,
    rows: Sequence[ResidualRow],
    *,
    table: str = "trading.greeks_residuals",
) -> int:
    if not rows:
        return 0
    dicts = [residual_to_ch_dict(r) for r in rows]
    cols = list(dicts[0].keys())
    data = [[d[c] for c in cols] for d in dicts]
    db, name = _split_table(table)
    client.insert(name, data, column_names=cols, database=db)
    return len(rows)


def _split_table(table: str) -> tuple[Optional[str], str]:
    if "." in table:
        db, name = table.split(".", 1)
        return db, name
    return None, table
