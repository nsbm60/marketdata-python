"""Local SQLite store for validation + residual rows (offline / pre-CH).

ClickHouse is the system of record for prod; this store mirrors the PR3 column
set for dry-runs and tests without a live CH. Optional dict mappers for CH insert.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from greeks.domain import FailureReason, JoinClass, OptionRight, RowStatus
from greeks.harness.rows import ResidualRow, ValidationRow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS greeks_validation (
    symbol TEXT NOT NULL,
    trade_ts TEXT NOT NULL,
    methodology_version TEXT NOT NULL,
    underlying TEXT NOT NULL,
    expiry TEXT NOT NULL,
    strike REAL NOT NULL,
    right TEXT NOT NULL,
    trade_price REAL NOT NULL,
    spot_at_trade REAL,
    forward REAL,
    discount REAL,
    time_to_expiry REAL,
    iv REAL,
    delta REAL,
    gamma REAL,
    vega REAL,
    theta REAL,
    rho REAL,
    status TEXT NOT NULL,
    reason_code TEXT,
    theta_interpretively_limited INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, trade_ts, methodology_version)
);

CREATE TABLE IF NOT EXISTS greeks_residuals (
    symbol TEXT NOT NULL,
    trade_ts TEXT NOT NULL,
    methodology_version TEXT NOT NULL,
    underlying TEXT NOT NULL,
    join_class TEXT NOT NULL,
    snapshot_ts TEXT,
    join_lag_capture_ms INTEGER,
    quote_ts TEXT,
    join_lag_quote_ms INTEGER,
    our_iv REAL,
    our_delta REAL,
    our_gamma REAL,
    our_vega REAL,
    our_theta REAL,
    our_rho REAL,
    vendor_iv REAL,
    vendor_delta REAL,
    vendor_gamma REAL,
    vendor_vega REAL,
    vendor_theta REAL,
    vendor_rho REAL,
    residual_iv_bps REAL,
    residual_delta REAL,
    residual_gamma REAL,
    residual_vega REAL,
    residual_theta REAL,
    residual_rho REAL,
    moneyness REAL,
    dte_years REAL,
    PRIMARY KEY (symbol, trade_ts, methodology_version)
);
"""


def _iso(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    return ts.astimezone(timezone.utc).isoformat()


class ResultsStore:
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

    def __enter__(self) -> ResultsStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def upsert_validation(self, rows: Sequence[ValidationRow]) -> int:
        n = 0
        for r in rows:
            self._conn.execute(
                """
                INSERT INTO greeks_validation (
                    symbol, trade_ts, methodology_version, underlying, expiry,
                    strike, right, trade_price, spot_at_trade, forward, discount,
                    time_to_expiry, iv, delta, gamma, vega, theta, rho,
                    status, reason_code, theta_interpretively_limited
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(symbol, trade_ts, methodology_version) DO UPDATE SET
                    status=excluded.status,
                    reason_code=excluded.reason_code,
                    iv=excluded.iv,
                    delta=excluded.delta,
                    gamma=excluded.gamma,
                    vega=excluded.vega,
                    theta=excluded.theta,
                    rho=excluded.rho,
                    forward=excluded.forward,
                    discount=excluded.discount,
                    time_to_expiry=excluded.time_to_expiry,
                    spot_at_trade=excluded.spot_at_trade
                """,
                (
                    r.symbol,
                    _iso(r.trade_ts),
                    r.methodology_version,
                    r.underlying,
                    r.expiry.isoformat(),
                    r.strike,
                    r.right.value,
                    r.trade_price,
                    r.spot_at_trade,
                    r.forward,
                    r.discount,
                    r.time_to_expiry,
                    r.iv,
                    r.delta,
                    r.gamma,
                    r.vega,
                    r.theta,
                    r.rho,
                    r.status.value,
                    r.reason_code.value if r.reason_code else None,
                    1 if r.theta_interpretively_limited else 0,
                ),
            )
            n += 1
        self._conn.commit()
        return n

    def upsert_residuals(self, rows: Sequence[ResidualRow]) -> int:
        n = 0
        for r in rows:
            self._conn.execute(
                """
                INSERT INTO greeks_residuals (
                    symbol, trade_ts, methodology_version, underlying, join_class,
                    snapshot_ts, join_lag_capture_ms, quote_ts, join_lag_quote_ms,
                    our_iv, our_delta, our_gamma, our_vega, our_theta, our_rho,
                    vendor_iv, vendor_delta, vendor_gamma, vendor_vega,
                    vendor_theta, vendor_rho,
                    residual_iv_bps, residual_delta, residual_gamma,
                    residual_vega, residual_theta, residual_rho,
                    moneyness, dte_years
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(symbol, trade_ts, methodology_version) DO UPDATE SET
                    join_class=excluded.join_class,
                    residual_iv_bps=excluded.residual_iv_bps,
                    residual_delta=excluded.residual_delta,
                    join_lag_capture_ms=excluded.join_lag_capture_ms,
                    join_lag_quote_ms=excluded.join_lag_quote_ms
                """,
                (
                    r.symbol,
                    _iso(r.trade_ts),
                    r.methodology_version,
                    r.underlying,
                    r.join_class.value,
                    _iso(r.snapshot_ts),
                    r.join_lag_capture_ms,
                    _iso(r.quote_ts),
                    r.join_lag_quote_ms,
                    r.our_iv,
                    r.our_delta,
                    r.our_gamma,
                    r.our_vega,
                    r.our_theta,
                    r.our_rho,
                    r.vendor_iv,
                    r.vendor_delta,
                    r.vendor_gamma,
                    r.vendor_vega,
                    r.vendor_theta,
                    r.vendor_rho,
                    r.residual_iv_bps,
                    r.residual_delta,
                    r.residual_gamma,
                    r.residual_vega,
                    r.residual_theta,
                    r.residual_rho,
                    r.moneyness,
                    r.dte_years,
                ),
            )
            n += 1
        self._conn.commit()
        return n

    def count_validation(
        self, *, methodology_version: Optional[str] = None
    ) -> dict[str, int]:
        sql = "SELECT status, COUNT(*) AS n FROM greeks_validation WHERE 1=1"
        params: list[object] = []
        if methodology_version is not None:
            sql += " AND methodology_version = ?"
            params.append(methodology_version)
        sql += " GROUP BY status"
        rows = self._conn.execute(sql, params).fetchall()
        return {str(r["status"]): int(r["n"]) for r in rows}

    def count_residuals_by_class(
        self, *, methodology_version: Optional[str] = None
    ) -> dict[str, int]:
        sql = "SELECT join_class, COUNT(*) AS n FROM greeks_residuals WHERE 1=1"
        params: list[object] = []
        if methodology_version is not None:
            sql += " AND methodology_version = ?"
            params.append(methodology_version)
        sql += " GROUP BY join_class"
        rows = self._conn.execute(sql, params).fetchall()
        return {str(r["join_class"]): int(r["n"]) for r in rows}

    def iter_validation(
        self, *, methodology_version: Optional[str] = None
    ) -> Iterator[ValidationRow]:
        sql = "SELECT * FROM greeks_validation WHERE 1=1"
        params: list[object] = []
        if methodology_version is not None:
            sql += " AND methodology_version = ?"
            params.append(methodology_version)
        for r in self._conn.execute(sql, params):
            yield _row_validation(r)

    def iter_residuals(
        self, *, methodology_version: Optional[str] = None
    ) -> Iterator[ResidualRow]:
        sql = "SELECT * FROM greeks_residuals WHERE 1=1"
        params: list[object] = []
        if methodology_version is not None:
            sql += " AND methodology_version = ?"
            params.append(methodology_version)
        for r in self._conn.execute(sql, params):
            yield _row_residual(r)

    def load_validation(
        self, *, methodology_version: Optional[str] = None
    ) -> list[ValidationRow]:
        return list(self.iter_validation(methodology_version=methodology_version))

    def load_residuals(
        self, *, methodology_version: Optional[str] = None
    ) -> list[ResidualRow]:
        return list(self.iter_residuals(methodology_version=methodology_version))


def validation_to_ch_dict(r: ValidationRow) -> dict[str, Any]:
    """Map to ClickHouse insert dict (PR3 column names)."""
    return {
        "symbol": r.symbol,
        "trade_ts": r.trade_ts,
        "methodology_version": r.methodology_version,
        "underlying": r.underlying,
        "expiry": r.expiry,
        "strike": r.strike,
        "right": r.right.value,
        "trade_price": r.trade_price,
        "spot_at_trade": r.spot_at_trade,
        "forward": r.forward,
        "discount": r.discount,
        "time_to_expiry": r.time_to_expiry,
        "iv": r.iv,
        "delta": r.delta,
        "gamma": r.gamma,
        "vega": r.vega,
        "theta": r.theta,
        "rho": r.rho,
        "status": r.status.value,
        "reason_code": r.reason_code.value if r.reason_code else None,
        "theta_interpretively_limited": 1 if r.theta_interpretively_limited else 0,
    }


def residual_to_ch_dict(r: ResidualRow) -> dict[str, Any]:
    return {
        "symbol": r.symbol,
        "trade_ts": r.trade_ts,
        "methodology_version": r.methodology_version,
        "underlying": r.underlying,
        "join_class": r.join_class.value,
        "snapshot_ts": r.snapshot_ts,
        "join_lag_capture_ms": r.join_lag_capture_ms,
        "quote_ts": r.quote_ts,
        "join_lag_quote_ms": r.join_lag_quote_ms,
        "our_iv": r.our_iv,
        "our_delta": r.our_delta,
        "our_gamma": r.our_gamma,
        "our_vega": r.our_vega,
        "our_theta": r.our_theta,
        "our_rho": r.our_rho,
        "vendor_iv": r.vendor_iv,
        "vendor_delta": r.vendor_delta,
        "vendor_gamma": r.vendor_gamma,
        "vendor_vega": r.vendor_vega,
        "vendor_theta": r.vendor_theta,
        "vendor_rho": r.vendor_rho,
        "residual_iv_bps": r.residual_iv_bps,
        "residual_delta": r.residual_delta,
        "residual_gamma": r.residual_gamma,
        "residual_vega": r.residual_vega,
        "residual_theta": r.residual_theta,
        "residual_rho": r.residual_rho,
        "moneyness": r.moneyness,
        "dte_years": r.dte_years,
    }


def _row_validation(r: sqlite3.Row) -> ValidationRow:
    rc = r["reason_code"]
    return ValidationRow(
        symbol=r["symbol"],
        trade_ts=datetime.fromisoformat(r["trade_ts"]),
        methodology_version=r["methodology_version"],
        underlying=r["underlying"],
        expiry=date.fromisoformat(r["expiry"]),
        strike=float(r["strike"]),
        right=OptionRight(r["right"]),
        trade_price=float(r["trade_price"]),
        spot_at_trade=r["spot_at_trade"],
        forward=r["forward"],
        discount=r["discount"],
        time_to_expiry=r["time_to_expiry"],
        iv=r["iv"],
        delta=r["delta"],
        gamma=r["gamma"],
        vega=r["vega"],
        theta=r["theta"],
        rho=r["rho"],
        status=RowStatus(r["status"]),
        reason_code=FailureReason(rc) if rc else None,
        theta_interpretively_limited=bool(r["theta_interpretively_limited"]),
    )


def _row_residual(r: sqlite3.Row) -> ResidualRow:
    return ResidualRow(
        symbol=r["symbol"],
        trade_ts=datetime.fromisoformat(r["trade_ts"]),
        methodology_version=r["methodology_version"],
        underlying=r["underlying"],
        join_class=JoinClass(r["join_class"]),
        snapshot_ts=(
            datetime.fromisoformat(r["snapshot_ts"]) if r["snapshot_ts"] else None
        ),
        join_lag_capture_ms=r["join_lag_capture_ms"],
        quote_ts=datetime.fromisoformat(r["quote_ts"]) if r["quote_ts"] else None,
        join_lag_quote_ms=r["join_lag_quote_ms"],
        our_iv=r["our_iv"],
        our_delta=r["our_delta"],
        our_gamma=r["our_gamma"],
        our_vega=r["our_vega"],
        our_theta=r["our_theta"],
        our_rho=r["our_rho"],
        vendor_iv=r["vendor_iv"],
        vendor_delta=r["vendor_delta"],
        vendor_gamma=r["vendor_gamma"],
        vendor_vega=r["vendor_vega"],
        vendor_theta=r["vendor_theta"],
        vendor_rho=r["vendor_rho"],
        residual_iv_bps=r["residual_iv_bps"],
        residual_delta=r["residual_delta"],
        residual_gamma=r["residual_gamma"],
        residual_vega=r["residual_vega"],
        residual_theta=r["residual_theta"],
        residual_rho=r["residual_rho"],
        moneyness=r["moneyness"],
        dte_years=r["dte_years"],
    )
