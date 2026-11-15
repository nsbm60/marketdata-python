-- Mirror of trading.option_trade in the Scala repo's db/database_schema.sql — the column contract the
-- greeks.history writer must match exactly (plus the greeks columns). Kept identical to the source.
CREATE TABLE IF NOT EXISTS trading.option_trade
(
    symbol                   LowCardinality(String),
    underlying               LowCardinality(String),
    session_date             Date,
    trade_ts                 DateTime64(3, 'UTC'),
    price                    Decimal64(4),
    size                     Float64,
    exchange                 Nullable(UInt16),
    conditions               Array(UInt16),
    sip_timestamp_ns         Int64,
    ordinal                  UInt32,
    sequence_number          Nullable(Int64),
    participant_timestamp_ns Nullable(Int64),
    correction               Nullable(UInt32),
    spot_at_trade            Nullable(Decimal64(4)),
    spot_trade_ts            Nullable(DateTime64(3, 'UTC')),
    quote_ts                 Nullable(DateTime64(3, 'UTC')),
    bid                      Nullable(Decimal64(4)),
    ask                      Nullable(Decimal64(4)),
    bid_size                 Nullable(UInt32),
    ask_size                 Nullable(UInt32),
    quote_lag_ms             Nullable(Int64),
    greeks_methodology       Nullable(String),
    greeks_basis             Nullable(String),
    greeks_status            Nullable(String),
    greeks_reason            Nullable(String),
    greeks_forward           Nullable(Float64),
    greeks_discount          Nullable(Float64),
    greeks_tte               Nullable(Float64),
    iv                       Nullable(Float64),
    delta                    Nullable(Float64),
    gamma                    Nullable(Float64),
    theta                    Nullable(Float64),
    vega                     Nullable(Float64),
    rho                      Nullable(Float64),
    greeks_computed_at       Nullable(DateTime64(3, 'UTC')),
    source                   LowCardinality(String) DEFAULT 'massive',
    ingested_at              DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMM(session_date)
ORDER BY (underlying, symbol, sip_timestamp_ns, ordinal)
SETTINGS index_granularity = 8192;
