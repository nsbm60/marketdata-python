# ClickHouse schema pointer

**System of record is not this repo.** This package never creates tables; the DDL is applied by hand.

| Artifact | Location |
|----------|----------|
| Canonical DDL | `/Users/nsm/Projects/Scala/MarketData/src/main/scala/com/nsbm/marketdata/db/database_schema.sql` |

## Tables used by iv_curve

| Table | Role |
|-------|------|
| `trading.iv_term_daily` | Source: one ATM IV per (underlying, day, expiry), method `atm_v1` (read only) |
| `trading.iv_curve_daily` | Output: the isotonic total-variance curve, method `iso_v1` |
| `trading.option_archive_ledger` | Progress ledger, dataset `iv_curve_iso_v1` |

The Python writer (`iv_curve.curve.COLUMNS`) writes every `trading.iv_curve_daily` column except
`computed_at` (the `ReplacingMergeTree` version DEFAULT). `tests/iv_curve/test_columns.py` reads the
canonical DDL above and asserts the column contract, as `tests/iv_term` does for `iv_term_daily`.
