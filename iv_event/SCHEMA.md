# ClickHouse schema pointer

**System of record is not this repo.** This package never creates tables; the DDL is applied by hand.

| Artifact | Location |
|----------|----------|
| Canonical DDL | `/Users/nsm/Projects/Scala/MarketData/src/main/scala/com/nsbm/marketdata/db/database_schema.sql` |

## Tables used by iv_event

| Table | Role |
|-------|------|
| `trading.iv_curve_daily` | Source: the isotonic total-variance curve, method `iso_v1` (read only) |
| `trading.rv_daily` | Source for the day weights: total realized variance, source `har-rv` (read only) |
| `trading.earnings_history` | Source: past earnings reaction sessions (read only) |
| `trading.earnings_calendar` | Source: future earnings dates (read only) |
| `trading.iv_event_window` | Output: one row per (underlying, day, window), method `fwd_v1` |
| `trading.iv_event_daytype` | Output: the measured gap→weight day-type table, method `fwd_v1` |
| `trading.option_archive_ledger` | Progress ledger, dataset `iv_event_fwd_v1` |

The Python writers (`iv_event.event.COLUMNS` for `iv_event_window`, `iv_event.daytype._DAYTYPE_COLUMNS`
for `iv_event_daytype`) write every column of their table except `computed_at` (the `ReplacingMergeTree`
version DEFAULT). `tests/iv_event/test_event_columns.py` reads the canonical DDL above and asserts both
column contracts, as `tests/iv_curve` does for `iv_curve_daily`.
