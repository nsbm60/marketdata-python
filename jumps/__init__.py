"""
Jump-day record (roadmap-2026-11-16 Stage 3b part 1): method bns_v1.

One row per trading.rv_daily row (source har-rv) in trading.jump_daily — per (symbol, trading_date),
whether an intraday jump and an overnight jump happened, their size, and the overnight move net of
the market. The intraday test is the Barndorff-Nielsen–Shephard RV-vs-BV ratio statistic
(Huang–Tauchen form); the overnight test is |return| against the trailing median of its own squared
returns (rescaled to a σ by the median of χ²₁, so Z_OVERNIGHT is in σ), raw and market-adjusted. The
adjustment is beta-based: overnight_ret − β·spy_ret, with β = cov(o, o_spy)/var(o_spy) estimated over
the name's trailing sessions (not β = 1). Every z is stored so the thresholds (a first setting) can
be tuned in Stage 3b part 3 against this record.

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
Z_OVERNIGHT = 4.0           # the overnight-jump cutoff, in σ (see MED_CHI2_1)
# The median of χ²₁: if X~N(0,1) then median(X²) = (Φ⁻¹(0.75))² ≈ 0.454936. The trailing median of
# overnight_ret² estimates median(o²) = MED_CHI2_1·σ², so σ² = median/MED_CHI2_1; dividing by it makes
# overnight_z a true σ multiple (without it the "4" cutoff was really ~2.7σ and over-flagged).
MED_CHI2_1 = NormalDist().inv_cdf(0.75) ** 2
ON_WINDOW = 66              # trailing sessions (with rv_daily rows) the overnight median is taken over
ON_MIN_HISTORY = 22         # fewer than this many recent sessions → overnight z is NULL
# Age limit, part of the definition: a trailing session counts only if it is within this many calendar
# days before the row's date. A name returning after a gap (e.g. SMCI, delisted 2018 → relisted Jan
# 2020) gets NULL until it has ON_MIN_HISTORY sessions this recent — a median built from year-old
# sessions is wrong. The trailing set is the previous rv_daily sessions, at most ON_WINDOW of them,
# and only those within ON_MAX_AGE_DAYS (excluding today).
ON_MAX_AGE_DAYS = 200

# Beta for the market adjustment: β = cov(o, o_spy)/var(o_spy) over the name's previous sessions on
# which both have a value, at most BETA_WINDOW of them and only within BETA_MAX_AGE_DAYS calendar days
# (excluding today). NULL below BETA_MIN_HISTORY such sessions or a zero var(o_spy). A longer window
# than the median's because a stable beta needs more history than a scale estimate.
BETA_WINDOW = 252
BETA_MIN_HISTORY = 66
BETA_MAX_AGE_DAYS = 400

# The statement's read lookback = the largest age limit, so each month reads everything the
# definition can use for its rows' trailing medians and betas (nothing older can count).
_JUMP_LOOKBACK_DAYS = max(ON_MAX_AGE_DAYS, BETA_MAX_AGE_DAYS)

# Fully-qualified table names (Python stores names, never a second copy of the DDL).
JUMP_DAILY_TABLE = "trading.jump_daily"
RV_DAILY_TABLE = "trading.rv_daily"
EARNINGS_HISTORY_TABLE = "trading.earnings_history"
