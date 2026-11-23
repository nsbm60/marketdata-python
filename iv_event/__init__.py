"""
Implied event detector (roadmap-2026-11-16 Stage 3): method fwd_v1.

For each (underlying, session_date) the iso_v1 total-variance curve (trading.iv_curve_daily) is split
into windows between expiry nodes. A window carries how much variance it holds beyond the name's normal
per-trading-day level, measured on a trading-day clock with weekend/holiday days weighted by how much
variance they actually fold in (trading.iv_event_daytype). Excess variance = a priced one-off move;
√excess is the implied ±move. Every window is stored (trading.iv_event_window) so the flag threshold
can be tuned later against realized jumps — this package only measures, it never flags a live decision.

- fwd.py      — the pure window math (node selection, effective days, normal/excess, earnings pick).
- daytype.py  — the measured day-weight calibration from rv_daily (the `daytype` command).
- event.py    — the ClickHouse aggregation over iv_curve_daily, the month/day processor, the ledger.
- show.py     — the read-only `show` and `top` inspection prints.
- __main__.py — the daytype / backfill / daily / show / top commands.

This package NEVER creates tables; the DDL for trading.iv_event_window and trading.iv_event_daytype
lives in Scala's db/database_schema.sql and is applied by hand. SCHEMA.md records the canonical path
for the column-contract test.
"""

# Method / provenance tags.
METHOD = "fwd_v1"                     # this package's method (iv_event_window.method, iv_event_daytype.method)
SOURCE_METHOD = "iso_v1"              # the iv_curve_daily method the windows are built from
LEDGER_DATASET = "iv_event_fwd_v1"    # trading.option_archive_ledger dataset marking processed pairs
RV_SOURCE = "har-rv"                  # the rv_daily source the day weights are measured from

# Stage 3b part 3 — the event-outcome record (iv_event_outcome.method) and the methods it reads.
OUTCOME_METHOD = "out_v1"             # this package's outcome method (iv_event_outcome.method)
WINDOW_METHOD = METHOD               # the detector method an outcome scores (iv_event_outcome.window_method)
JUMP_METHOD = "bns_v1"               # the jump_daily method the outcome reads
# A date is market-wide when at least this many names carry overnight_jump_adj = 1 that day — the SAME
# constant and definition as the Scala EarningsReaction reaction rule (MARKET_WIDE_MIN_NAMES); keep
# the two in step.
MARKET_WIDE_MIN_NAMES = 100
# The outcome backfill-summary prints break windows into excess_days bands and require this much
# window support before a band is reported (as `top`/`show` use SHOW_FLAG_SUPPORT).
OUTCOME_SUMMARY_MIN_SUPPORT = 100.0
# excess_days bands for proof item 2: [lo, hi) in normal-day units; the last is open-ended.
OUTCOME_EXCESS_DAYS_BANDS = ((None, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, None))

# The curve rows the windows read (iv_curve_daily's own kind/status vocabulary).
CURVE_KIND_EXPIRY = "expiry"
CURVE_STATUS_OK = "ok"

# Named thresholds — all here, nothing defaulted elsewhere.
MIN_NODE_SUPPORT = 50.0       # an expiry becomes a node only with at least this much support
MIN_WINDOWS = 3               # fewer windows than this → one insufficient_windows row, no excess
MIN_DAYTYPE_N = 200           # a gap that occurs with fewer rv_total observations than this is an error
GAP_BASE = 1                  # the gap whose median defines weight 1
GAP_MAX_BUCKET = 5            # gaps of 5+ calendar days pool into one bucket, stored as gap_days = 5
SHOW_FLAG_EXCESS_DAYS = 2.0   # `show` marks a window EVENT at/above this excess_days …
SHOW_FLAG_SUPPORT = 100.0     # … and this support
TOP_DEFAULT_MIN_SUPPORT = 100.0
TRADING_DAYS_PER_YEAR = 252   # annualisation base for the vol prints

# This package's own row statuses and earnings-source tags.
STATUS_OK = "ok"
STATUS_INSUFFICIENT_WINDOWS = "insufficient_windows"
EARNINGS_HISTORY = "earnings_history"
EARNINGS_HISTORY_UNRESOLVED = "earnings_history_unresolved"  # reaction day unknown → reported day OR next trading day
EARNINGS_CALENDAR = "earnings_calendar"

# Fully-qualified table names (Python stores names, never a second copy of the DDL).
IV_EVENT_TABLE = "trading.iv_event_window"
IV_EVENT_DAYTYPE_TABLE = "trading.iv_event_daytype"
IV_EVENT_OUTCOME_TABLE = "trading.iv_event_outcome"
IV_CURVE_TABLE = "trading.iv_curve_daily"
RV_DAILY_TABLE = "trading.rv_daily"
JUMP_DAILY_TABLE = "trading.jump_daily"
CORPORATE_ACTION_TABLE = "trading.corporate_action"
SPLIT_EFFECTIVE_TABLE = "trading.split_effective"
EARNINGS_HISTORY_TABLE = "trading.earnings_history"
EARNINGS_CALENDAR_TABLE = "trading.earnings_calendar"
LEDGER_TABLE = "trading.option_archive_ledger"
