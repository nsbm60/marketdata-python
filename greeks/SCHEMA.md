# ClickHouse schema pointer

**System of record is not this repo.**

| Artifact | Location |
|----------|----------|
| Canonical DDL | `/Users/nsm/Projects/Scala/MarketData/src/main/scala/com/nsbm/marketdata/db/database_schema.sql` |
| Migrations | `/Users/nsm/Projects/Scala/MarketData/migrations/` |
| PR3 migration | `migrations/greeks_validation_tables.sql` |

## Tables used by greeks validation

| Table | Role | DDL status |
|-------|------|------------|
| `trading.option_snapshot` | Vendor greeks baseline for harness join | Exists |
| `trading.option_contract` | Contract reference | Exists |
| `trading.earnings_calendar` | Earnings-window residual slice | Exists |
| `trading.greeks_validation` | Per-trade IV/greeks (ours) | **PR3** — schema + migration added |
| `trading.greeks_residuals` | Harness join residuals | **PR3** — schema + migration added |
| `trading.sofr_daily` | SOFR by date for carry | **PR3** — schema + migration added |

Do not maintain a second copy of CREATE TABLE under `marketdata-python/`. Python config only stores fully-qualified table names (`config/greeks.yaml` → `tables.*`).

## Column summary (PR3)

### `trading.sofr_daily`
`date`, `rate` (decimal), `source`, `fetched_at` — `ReplacingMergeTree(fetched_at)`.

### `trading.greeks_validation`
Key: `(symbol, trade_ts, methodology_version)`.  
Includes `spot_at_trade`, `forward`, `discount`, `time_to_expiry`, IV/greeks, `status`, `reason_code`.  
Timestamps: `DateTime64(3, 'UTC')` only.

### `trading.greeks_residuals`
Join class + **two lag clocks** (capture `snapshot_ts` / `join_lag_capture_ms`, quote `quote_ts` / `join_lag_quote_ms`), our vs vendor greeks, residuals (IV in vol bps).

**Prod apply:** review `migrations/greeks_validation_tables.sql` then run against ClickHouse (not auto-applied from Python).
