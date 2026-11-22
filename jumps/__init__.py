"""
Jump-day record (roadmap-2026-11-16 Stage 3b part 1): method bns_v1.

One row per trading.rv_daily row (source har-rv) in trading.jump_daily — per (symbol, trading_date),
whether an intraday jump and an overnight jump happened, their size, and the overnight move net of
the market (SPY). The intraday test is the Barndorff-Nielsen–Shephard RV-vs-BV ratio statistic
(Huang–Tauchen form); the overnight test is |return| against the trailing median of its own squared
returns, raw and SPY-adjusted. Every z is stored so the thresholds (a first setting) can be tuned in
Stage 3b part 3 against this record.

- jump_daily.py — the set-based SQL (one statement per month like rv_daily), the fail-fast checks,
  the backfill summaries, and the read-only `show`.
- measures.py   — the pure-Python oracle for the z statistics and flags (unit-tested, no ClickHouse).
- __main__.py   — the backfill / daily / show commands.

This package NEVER creates tables; the trading.jump_daily DDL lives in Scala's db/database_schema.sql
and is applied by hand. SCHEMA.md records the canonical path for the column-contract test. The
overnight return is NOT copied from rv_daily — it is recomputed through rv_daily's own `_with_core`
and `OVERNIGHT_RET_SQL`, so it is the same return rv_daily squares.
"""

from __future__ import annotations

import math
from statistics import NormalDist

# Method / provenance tags.
METHOD = "bns_v1"            # this package's method (jump_daily.method)
RV_SOURCE = "har-rv"         # the rv_daily source the rows are drawn from
SPY_SYMBOL = "SPY"           # the market proxy subtracted for the adjusted overnight move

# Named thresholds — all here, nothing defaulted elsewhere. A first setting; Stage 3b part 3 tunes
# them against the stored z's, which is why every z is persisted.
# θ = (π/2)² + π − 5 ≈ 0.6090 — the BNS asymptotic variance constant of the (RV−BV)/RV ratio statistic.
THETA = (math.pi / 2) ** 2 + math.pi - 5
# Φ⁻¹(0.999) ≈ 3.0902 — the intraday-jump cutoff on the (one-sided) standard-normal ratio statistic.
Z_INTRADAY = NormalDist().inv_cdf(0.999)
Z_OVERNIGHT = 4.0           # the overnight-jump cutoff on |overnight return| / trailing-median scale
ON_WINDOW = 66              # trailing sessions (with rv_daily rows) the overnight median is taken over
ON_MIN_HISTORY = 22         # fewer than this many previous sessions → overnight z is NULL
# Calendar lookback each month's statement reads so the ON_WINDOW trailing median is complete: ~138
# trading sessions, comfortably over ON_WINDOW. A per-month guard fails by name/date if this is ever
# too short to hold a row's ON_WINDOW history (so a median is never silently taken over fewer rows).
_JUMP_LOOKBACK_DAYS = 200

# Fully-qualified table names (Python stores names, never a second copy of the DDL).
JUMP_DAILY_TABLE = "trading.jump_daily"
RV_DAILY_TABLE = "trading.rv_daily"
EARNINGS_HISTORY_TABLE = "trading.earnings_history"
