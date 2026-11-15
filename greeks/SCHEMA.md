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

### Overnight discount rate (SOFR + pre-SOFR T-bill)
SOFR (`source='nyfed'`) begins **2018-04-02**, but historical greeks price back to 2016-01-04. One
overnight series per session feeds the ACT/360 accrual (`continuous_rate_from_sofr`):

- a date **on or after** the first SOFR fix (read from the table, never hard-coded) uses that date's
  SOFR fix; a missing fix there is an error — the T-bill never fills a post-SOFR gap;
- a date **before** the first SOFR fix uses that date's **1-month Treasury-bill yield** from
  `trading.treasury_yields` (`source='massive'`, `y_1m`, stored decimal), converted to the same
  ACT/360 simple overnight basis the SOFR accrual uses:

  ```
  r_360 = y_365 * 360 / 365
  ```

  (Treasury yields are ACT/365; the accrual compounds `(1 + r/360)` per day.) A pre-SOFR date with no
  1-month yield is an error naming the date — nothing is defaulted.

A window crossing 2018-04-02 compounds T-bills before the boundary and SOFR after it. The loader logs
the source once per session as `rate=TBILL1M`, `rate=SOFR`, or `rate=TBILL1M+SOFR`. Live greeks only
request recent dates, so the live server only ever sees SOFR. No new column.

### `trading.greeks_validation`
Key: `(symbol, trade_ts, methodology_version)`.  
Includes `spot_at_trade`, `forward`, `discount`, `time_to_expiry`, IV/greeks, `status`, `reason_code`.  
Timestamps: `DateTime64(3, 'UTC')` only.

### `trading.greeks_residuals`
Join class + **two lag clocks** (capture `snapshot_ts` / `join_lag_capture_ms`, quote `quote_ts` / `join_lag_quote_ms`), our vs vendor greeks, residuals (IV in vol bps).

**Prod apply:** review `migrations/greeks_validation_tables.sql` then run against ClickHouse (not auto-applied from Python).
