"""
Daily implied total-variance curve on a trading-day clock (roadmap-2026-11-16 Appendix A / Stage 2;
pc-objective-function-draft.md §6–§7).

iv_term_daily gives one ATM IV per (underlying, day, expiry), but single expiries are noisy — thin
ones wander and neighbours can imply negative forward variance. This builds one clean curve per
(underlying, session_date): total variance fitted against TRADING DAYS to expiry with a weighted
isotonic (non-decreasing) regression anchored at (0, 0), weighted by trade count. Each stored point
carries a `support` number (Σ n_trades of its pooled block) that later becomes the model's sample
weight. Thin data gets low weight, never a guessed value.

- iso.py      — the weighted pool-adjacent-violators fit + horizon interpolation (pure NumPy).
- curve.py    — the ClickHouse aggregation over iv_term_daily, the month/day processor, the ledger.
- show.py     — the read-only `show` and `coverage` inspection prints.
- __main__.py — the backfill / daily / show / coverage commands.

The table lives in Scala's db/database_schema.sql (trading.iv_curve_daily); this package NEVER creates
it. SCHEMA.md records the canonical path for the column-contract test.
"""

# Method / provenance tags.
METHOD = "iso_v1"
SOURCE_METHOD = "atm_v1"            # the iv_term_daily method this curve is built from
LEDGER_DATASET = "iv_curve_iso_v1"  # trading.option_archive_ledger dataset marking processed pairs

# The fixed horizons (trading days) at which the fitted curve is sampled. Named here, nowhere else.
HORIZONS_TD = (5, 10, 21, 42)

TRADING_DAYS_PER_YEAR = 252  # annualisation base for the forward-vol print in `show`

# Row kinds and statuses.
KIND_EXPIRY = "expiry"
KIND_HORIZON = "horizon"
STATUS_OK = "ok"
STATUS_BEYOND_LAST_EXPIRY = "beyond_last_expiry"

# Fully-qualified table names (Python stores names, never a second copy of the DDL).
IV_CURVE_TABLE = "trading.iv_curve_daily"
IV_TERM_TABLE = "trading.iv_term_daily"
LEDGER_TABLE = "trading.option_archive_ledger"
