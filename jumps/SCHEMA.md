# ClickHouse schema pointer

**System of record is not this repo.** This package never creates tables; the DDL is applied by hand.

| Artifact | Location |
|----------|----------|
| Canonical DDL | `/Users/nsm/Projects/Scala/MarketData/src/main/scala/com/nsbm/marketdata/db/database_schema.sql` |

## Tables used by jumps

| Table | Role |
|-------|------|
| `trading.rv_daily` | Source: realized + bipower variance and quarticity, source `har-rv` (read only) |
| `trading.earnings_history` | Source: past earnings reported dates, for the backfill-summary join (read only) |
| `trading.jump_daily` | Output: one row per rv_daily row, method `bns_v1` |

The Python writer (`jumps.jump_daily.COLUMNS`) writes every column of `trading.jump_daily`, including
`computed_at` — unlike `iv_event`, the `jump_daily` DDL has no DEFAULT on `computed_at`, so the
`INSERT … SELECT` supplies it (`now64(3)`). `tests/jumps/test_jump_columns.py` reads the canonical DDL
above and asserts the column contract, as `tests/iv_event/test_event_columns.py` does for its tables.
