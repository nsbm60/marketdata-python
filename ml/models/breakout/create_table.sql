-- ClickHouse table for breakout candidates
-- Run this once to create the table:
--   clickhouse-client --query "$(cat ml/models/breakout/create_table.sql)"
-- Or via Python:
--   ch.command(open('ml/models/breakout/create_table.sql').read())

CREATE TABLE IF NOT EXISTS trading.breakout_candidates (
    -- Identity
    symbol           String,
    ts               DateTime64(3, 'America/New_York'),
    timeframe        UInt8,        -- 5 for 5-minute bars

    -- Signal details
    direction        Enum8('long' = 1, 'short' = 2),
    conviction       Enum8('high' = 1, 'medium' = 2, 'low' = 3),

    -- Level context
    level_broken     Float64,
    level_type       String,       -- 'intraday_high', 'intraday_low', 'prior_close'
    level_age_min    UInt32,
    level_test_count UInt8,

    -- Ribbon context
    ribbon_state     String,
    ribbon_spread    Float64,
    ribbon_age_bars  UInt16,
    gap_induced      UInt8,

    -- Break bar metrics
    break_bar_atr    Float64,
    close_position   Float64,
    volume_ratio     Float64,

    -- Clear air
    next_level       Float64,
    clear_air_atr    Float64,

    -- Condition tracking
    conditions_met   UInt8,
    failed_conditions String,

    -- Would have published (for analysis)
    published        UInt8,

    -- Outcome (filled at EOD)
    outcome_30min    Nullable(Float64),
    outcome_60min    Nullable(Float64),
    outcome_eod      Nullable(Float64)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (symbol, ts);
