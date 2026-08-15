#!/usr/bin/env python3
"""Scan trading.option_snapshot for validation-window candidates.

Emits per (underlying, date) quality metrics, then proposes contiguous
date ranges per underlying that meet acceptance thresholds.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import clickhouse_connect

# Repo root on path for discovery / shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discovery.service_locator import ServiceLocator

DATABASE = os.environ.get("CLICKHOUSE_DATABASE", "trading")
TABLE = "option_snapshot"

# trading.option_snapshot columns (see DESCRIBE TABLE option_snapshot)
COLS = {
    "symbol": "symbol",
    "underlying": "underlying",
    "ts": "timestamp",
    "bid": "bid",
    "ask": "ask",
    "iv": "iv",
    "delta": "delta",
    "vega": "vega",
}

CONTRACT_TABLE = "option_contract"

# Standard OSI: ROOT(1-6 letters) + YYMMDD + C|P + strike*1000 (8 digits)
OSI_RE = r"^[A-Z]{1,6}[0-9]{6}[CP][0-9]{8}$"

RIGHT_CALL = "'C'"

# Acceptance thresholds for a (underlying, date) to count as "clean"
MIN_ROWS_PER_DAY = 5_000        # density floor; lower for quiet names
MAX_CONTAM_PCT = 1.0            # % rows with zero IV despite nonzero bid
MIN_TWO_SIDED_PCT = 90.0        # % rows with bid > 0 and ask > 0
MIN_VALID_IV_PCT = 90.0         # % rows with iv > 0 and |delta| <= 1
MIN_RUN_DAYS = 10               # shortest contiguous run worth reporting

# ---------------------------------------------------------------- query

DAILY_SQL = f"""
WITH parsed AS (
    SELECT
        s.{COLS['underlying']}          AS underlying,
        c.expiration_date               AS expiry,
        c.strike_price                  AS strike,
        toString(c.call_put)            AS opt_right,
        toDate(s.{COLS['ts']})          AS d,
        toDate(s.{COLS['ts']}) = c.expiration_date AS is_expiry_day,
        s.{COLS['bid']}                 AS bid,
        s.{COLS['ask']}                 AS ask,
        s.{COLS['iv']}                  AS iv,
        s.{COLS['delta']}               AS delta,
        s.{COLS['vega']}                AS vega
    FROM {DATABASE}.{TABLE} AS s
    INNER JOIN {DATABASE}.{CONTRACT_TABLE} AS c
        ON s.{COLS['symbol']} = c.option_symbol
)
SELECT
    underlying,
    d,
    count()                                                     AS rows,
    uniqExact(expiry, strike, opt_right)                        AS contracts,
    uniqExact(expiry)                                           AS expiries,
    round(100 * countIf(bid > 0 AND ask > 0) / count(), 2)      AS two_sided_pct,
    round(100 * countIf(iv > 0 AND abs(delta) <= 1 AND vega >= 0)
          / count(), 2)                                         AS valid_iv_pct,
    round(100 * countIf(iv = 0 AND bid > 0) / count(), 2)       AS contam_pct,
    round(100 * countIf(iv = 0 AND bid > 0 AND opt_right = {RIGHT_CALL})
          / greatest(countIf(opt_right = {RIGHT_CALL}), 1), 2)  AS contam_call_pct,
    round(100 * countIf(iv = 0 AND bid > 0 AND opt_right != {RIGHT_CALL})
          / greatest(countIf(opt_right != {RIGHT_CALL}), 1), 2) AS contam_put_pct,
    countIf(is_expiry_day)                                      AS expiry_day_rows
FROM parsed
GROUP BY underlying, d
ORDER BY underlying, d
"""

# ---------------------------------------------------------------- logic


def connect_clickhouse():
    """
    Connect to ClickHouse via ZMQ service discovery (repo-canonical), unless
    CLICKHOUSE_HOST is set explicitly (useful off-network).
    """
    host = os.environ.get("CLICKHOUSE_HOST")
    port = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
    if host:
        print(f"ClickHouse (explicit) {host}:{port}")
    else:
        endpoint = ServiceLocator.wait_for_service(
            service_name=ServiceLocator.CLICKHOUSE,
            timeout_sec=30,
        )
        host, port = endpoint.host, endpoint.port
        print(f"ClickHouse discovered at {host}:{port} "
              f"(set CLICKHOUSE_HOST/PORT to override)")
    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", "Aector99"),
        database=DATABASE,
    )


@dataclass(frozen=True)
class DayStat:
    underlying: str
    d: object
    rows: int
    contracts: int
    expiries: int
    two_sided_pct: float
    valid_iv_pct: float
    contam_pct: float
    contam_call_pct: float
    contam_put_pct: float
    expiry_day_rows: int

    @property
    def clean(self) -> bool:
        return (
            self.rows >= MIN_ROWS_PER_DAY
            and self.contam_pct <= MAX_CONTAM_PCT
            and self.two_sided_pct >= MIN_TWO_SIDED_PCT
            and self.valid_iv_pct >= MIN_VALID_IV_PCT
        )


def contiguous_runs(days: list[DayStat]) -> list[tuple[DayStat, DayStat, int, int]]:
    """Contiguous-by-calendar clean runs: (start, end, n_days, expiry_day_rows)."""
    runs: list[tuple[DayStat, DayStat, int, int]] = []
    start: DayStat | None = None
    prev: DayStat | None = None
    exp_rows = 0
    for day in days:
        if day.clean:
            if start is None:
                start, exp_rows = day, 0
            exp_rows += day.expiry_day_rows
            prev = day
        else:
            if start is not None and prev is not None:
                n = (prev.d - start.d).days + 1
                if n >= MIN_RUN_DAYS:
                    runs.append((start, prev, n, exp_rows))
            start, prev = None, None
    if start is not None and prev is not None:
        n = (prev.d - start.d).days + 1
        if n >= MIN_RUN_DAYS:
            runs.append((start, prev, n, exp_rows))
    return runs


def assert_osi_compliance(client) -> None:
    """Fail loud if any non-OSI symbols exist — do not silently filter them."""
    r = client.query(f"""
        SELECT
            countIf(NOT match({COLS['symbol']}, '{OSI_RE}')) AS non_osi_rows,
            uniqExactIf({COLS['symbol']}, NOT match({COLS['symbol']}, '{OSI_RE}')) AS non_osi_symbols,
            minIf(toDate({COLS['ts']}), NOT match({COLS['symbol']}, '{OSI_RE}')) AS first_bad_day,
            maxIf(toDate({COLS['ts']}), NOT match({COLS['symbol']}, '{OSI_RE}')) AS last_bad_day
        FROM {DATABASE}.{TABLE}
    """)
    bad_rows, bad_syms, first_day, last_day = r.result_rows[0]
    if bad_rows:
        print(
            f"ERROR: {bad_rows:,} non-OSI rows ({bad_syms:,} symbols) in {DATABASE}.{TABLE} "
            f"({first_day} .. {last_day}).\n"
            "Run tools/audit_option_snapshot_symbols.py for detail, then apply "
            "migrations/fix_hierarchical_symbols.sql before scanning validation windows.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> int:
    client = connect_clickhouse()
    assert_osi_compliance(client)
    result = client.query(DAILY_SQL)
    stats = [DayStat(*row) for row in result.result_rows]

    by_underlying: dict[str, list[DayStat]] = {}
    for s in stats:
        by_underlying.setdefault(s.underlying, []).append(s)

    print(f"{'UNDERLYING':<8} {'DATE':<12} {'ROWS':>9} {'CNTRCT':>7} {'EXP':>4} "
          f"{'2SIDED%':>8} {'VALIDIV%':>9} {'CONTAM%':>8} "
          f"{'C.CALL%':>8} {'C.PUT%':>7} {'EXPDAY':>7}  FLAG")
    for u, days in sorted(by_underlying.items()):
        for s in days:
            flag = "" if s.clean else "  <-- dirty"
            print(f"{s.underlying:<8} {s.d!s:<12} {s.rows:>9} {s.contracts:>7} "
                  f"{s.expiries:>4} {s.two_sided_pct:>8} {s.valid_iv_pct:>9} "
                  f"{s.contam_pct:>8} {s.contam_call_pct:>8} "
                  f"{s.contam_put_pct:>7} {s.expiry_day_rows:>7}{flag}")

    print("\n=== Candidate windows (clean contiguous runs, "
          f">= {MIN_RUN_DAYS} calendar days) ===")
    any_run = False
    for u, days in sorted(by_underlying.items()):
        for start, end, n, exp_rows in contiguous_runs(days):
            any_run = True
            print(f"{u:<8} {start.d} .. {end.d}  ({n:>3}d)  "
                  f"expiry-day rows: {exp_rows}")
    if not any_run:
        print("none — loosen thresholds or inspect the daily table above")
    return 0


if __name__ == "__main__":
    sys.exit(main())