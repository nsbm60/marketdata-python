"""Local SQLite staging for pulled option trades + spot_at_trade.

Prefer staging → invert (PR5) over writing ``greeks_validation`` at pull time.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Sequence

from greeks.pull.alpaca_spot import SpotAtTrade
from greeks.pull.massive_trades import OptionTradePrint

_SCHEMA = """
CREATE TABLE IF NOT EXISTS staged_trades (
    symbol            TEXT NOT NULL,
    trade_ts          TEXT NOT NULL,
    session_date      TEXT NOT NULL,
    underlying        TEXT NOT NULL,
    price             REAL NOT NULL,
    size              REAL NOT NULL,
    exchange          INTEGER,
    conditions        TEXT,
    sequence_number   INTEGER,
    sip_timestamp_ns  INTEGER NOT NULL,
    spot_at_trade     REAL,
    spot_trade_ts     TEXT,
    PRIMARY KEY (symbol, sip_timestamp_ns, sequence_number)
);
CREATE INDEX IF NOT EXISTS idx_staged_session
    ON staged_trades(underlying, session_date);
"""


@dataclass(frozen=True)
class StagedTrade:
    symbol: str
    trade_ts: datetime
    session_date: date
    underlying: str
    price: float
    size: float
    exchange: Optional[int]
    conditions: str
    sequence_number: Optional[int]
    sip_timestamp_ns: int
    spot_at_trade: Optional[float]
    spot_trade_ts: Optional[datetime]


class TradeStaging:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=60.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TradeStaging:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def upsert_trades(
        self,
        session_date: date,
        underlying: str,
        pairs: Sequence[tuple[OptionTradePrint, Optional[SpotAtTrade]]],
    ) -> int:
        """Insert or replace staged rows. Returns rows written."""
        n = 0
        und = underlying.upper()
        for ot, spot in pairs:
            cond = ",".join(str(c) for c in ot.conditions)
            seq = ot.sequence_number if ot.sequence_number is not None else -1
            self._conn.execute(
                """
                INSERT INTO staged_trades (
                    symbol, trade_ts, session_date, underlying, price, size,
                    exchange, conditions, sequence_number, sip_timestamp_ns,
                    spot_at_trade, spot_trade_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, sip_timestamp_ns, sequence_number) DO UPDATE SET
                    price=excluded.price,
                    size=excluded.size,
                    spot_at_trade=excluded.spot_at_trade,
                    spot_trade_ts=excluded.spot_trade_ts
                """,
                (
                    ot.symbol,
                    ot.trade_ts.astimezone(timezone.utc).isoformat(),
                    session_date.isoformat(),
                    und,
                    ot.price,
                    ot.size,
                    ot.exchange,
                    cond,
                    seq,
                    ot.sip_timestamp_ns,
                    spot.spot if spot is not None else None,
                    (
                        spot.spot_trade_ts.astimezone(timezone.utc).isoformat()
                        if spot is not None
                        else None
                    ),
                ),
            )
            n += 1
        self._conn.commit()
        return n

    def count(
        self, *, underlying: Optional[str] = None, session_date: Optional[date] = None
    ) -> int:
        sql = "SELECT COUNT(*) AS n FROM staged_trades WHERE 1=1"
        params: list[object] = []
        if underlying is not None:
            sql += " AND underlying = ?"
            params.append(underlying.upper())
        if session_date is not None:
            sql += " AND session_date = ?"
            params.append(session_date.isoformat())
        row = self._conn.execute(sql, params).fetchone()
        return int(row["n"]) if row else 0

    def iter_session(
        self, underlying: str, session_date: date
    ) -> Iterator[StagedTrade]:
        rows = self._conn.execute(
            """
            SELECT * FROM staged_trades
            WHERE underlying = ? AND session_date = ?
            ORDER BY sip_timestamp_ns ASC
            """,
            (underlying.upper(), session_date.isoformat()),
        )
        for r in rows:
            yield _row_to_staged(r)


def _row_to_staged(r: sqlite3.Row) -> StagedTrade:
    spot_ts = r["spot_trade_ts"]
    return StagedTrade(
        symbol=r["symbol"],
        trade_ts=datetime.fromisoformat(r["trade_ts"]),
        session_date=date.fromisoformat(r["session_date"]),
        underlying=r["underlying"],
        price=float(r["price"]),
        size=float(r["size"]),
        exchange=r["exchange"],
        conditions=r["conditions"] or "",
        sequence_number=r["sequence_number"],
        sip_timestamp_ns=int(r["sip_timestamp_ns"]),
        spot_at_trade=(
            float(r["spot_at_trade"]) if r["spot_at_trade"] is not None else None
        ),
        spot_trade_ts=datetime.fromisoformat(spot_ts) if spot_ts else None,
    )
