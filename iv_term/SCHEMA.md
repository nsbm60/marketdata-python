# ClickHouse schema pointer

**System of record is not this repo.** This package never creates tables; the DDL is applied by hand.

| Artifact | Location |
|----------|----------|
| Canonical DDL | `/Users/nsm/Projects/Scala/MarketData/src/main/scala/com/nsbm/marketdata/db/database_schema.sql` |

## Tables used by iv_term

| Table | Role |
|-------|------|
| `trading.option_trade` | Source: per-trade v2 trade-basis greeks (read only) |
| `trading.iv_term_daily` | Output: one ATM term-structure row per (method, underlying, session_date, expiry) |
| `trading.option_archive_ledger` | Progress ledger, dataset `iv_term_atm_v1` |

The Python writer (`iv_term.term.COLUMNS`) writes every `trading.iv_term_daily` column except
`computed_at` (the `ReplacingMergeTree` version DEFAULT). `tests/iv_term/test_columns.py` reads the
canonical DDL above and asserts the column contract, as `tests/greeks` does for `option_trade`.
