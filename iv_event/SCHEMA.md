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
| `trading.jump_daily` | Source for the outcome record: the jump-day record, method `bns_v1` (read only) |
| `trading.corporate_action` | Source for the outcome record: corporate actions (read only) |
| `trading.split_effective` | Source for the outcome record: effective splits view (read only) |
| `trading.iv_event_window` | Output: one row per (underlying, day, window), method `fwd_v1` |
| `trading.iv_event_daytype` | Output: the measured gap→weight day-type table, method `fwd_v1` |
| `trading.iv_event_outcome` | Output: one row per passed window, method `out_v1` (what happened in it) |
| `trading.option_archive_ledger` | Progress ledger, dataset `iv_event_fwd_v1` |

The Python writers (`iv_event.event.COLUMNS` for `iv_event_window`, `iv_event.daytype._DAYTYPE_COLUMNS`
for `iv_event_daytype`) write every column of their table except `computed_at` (the `ReplacingMergeTree`
version DEFAULT). `iv_event.outcome_event.OUTCOME_COLUMNS` writes every column of `iv_event_outcome`
**including `computed_at`** (its DDL has no DEFAULT on it, so the writer supplies `now()`, as
`jump_daily` does). `tests/iv_event/test_event_columns.py` reads the canonical DDL above and asserts all
three column contracts, as `tests/iv_curve` does for `iv_curve_daily`. `iv_event.outcome_event`
never creates or alters the outcome table; `_require_outcome_table` fails fast if it or any column is
missing (the operator applies the DDL).
