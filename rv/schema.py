"""
DDL for the HAR tables, mirroring Scala's db/database_schema.sql (the canonical schema). Kept here
so a fresh run can create the tables; any change must be made in both places.
"""

RV_DAILY_DDL = """
CREATE TABLE IF NOT EXISTS trading.rv_daily
(
    symbol         LowCardinality(String),
    trading_date   Date,
    rv_intraday    Float64,
    rv_overnight   Float64,
    rv_total       Float64,
    n_bars         UInt16,
    expected_bars  UInt16,
    split_adjusted UInt8,
    source         LowCardinality(String) DEFAULT 'har-rv',
    computed_at    DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (symbol, trading_date)
"""

RV_FORECAST_DDL = """
CREATE TABLE IF NOT EXISTS trading.rv_forecast
(
    symbol         LowCardinality(String),
    as_of_date     Date,
    horizon_days   UInt16,
    sigma_forecast Float64,
    model_version  LowCardinality(String),
    fit_date       Date,
    rv_d           Float64,
    rv_w           Float64,
    rv_m           Float64,
    computed_at    DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (model_version, symbol, as_of_date, horizon_days)
"""


def ensure_tables(ch) -> None:
    ch.command(RV_DAILY_DDL)
    ch.command(RV_FORECAST_DDL)
