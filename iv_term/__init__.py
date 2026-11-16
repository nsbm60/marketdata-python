"""
Daily ATM term-structure snapshot — the implied side of the volatility model
(Scala/MarketData docs/specs/pc-objective-function-draft.md §6–§7).

For each (underlying, session_date, expiry) this builds one robust at-the-money IV and its total
variance (IV²·T) from the day's last hour of trades (15:00–16:00 ET), on the TRADE-basis greeks in
trading.option_trade. Everything downstream (labels, inputs, forward variance between expiries)
reads trading.iv_term_daily, not option_trade.

- fit.py      — the weighted quadratic smile fit and the status ladder (pure NumPy, no ClickHouse).
- term.py     — the ClickHouse aggregation, the month/day processor, the ledger and restart-skip.
- show.py     — the read-only term-structure inspection print.
- __main__.py — the backfill / daily / show commands.

The table lives in Scala's db/database_schema.sql (trading.iv_term_daily); this package NEVER creates
it — the DDL is applied by hand (greeks/SCHEMA.md posture). SCHEMA.md records the canonical path.

Method atm_v1: per group, fit iv = a + b·k + c·k² weighted by trade size, k = ln(strike/forward),
over the 15:00–16:00 ET prints with |delta| in [DELTA_LOW, DELTA_HIGH]. ATM IV = a; total_var = a²·tte.
"""

# Method / provenance tags, written into every row.
METHOD = "atm_v1"
GREEKS_METHODOLOGY = "v2"  # only v2 trade-basis greeks are read
LEDGER_DATASET = "iv_term_atm_v1"  # trading.option_archive_ledger dataset marking processed (day, underlying)

# Eligibility and fit constants — the whole method's knobs, in one place; nothing is defaulted
# elsewhere and there is no fallback to a wider window or band.
MIN_TRADES = 5     # fewer qualifying rows -> status too_few_trades
MIN_STRIKES = 3    # fewer distinct strikes -> status too_few_strikes
DELTA_LOW = 0.25   # |delta| band, inclusive, calls and puts both
DELTA_HIGH = 0.75
WINDOW_ET_HOUR = 15  # trades whose ET hour == 15, i.e. 15:00:00 inclusive to 16:00:00 exclusive
# Condition-number ceiling for the SCALED normal-equations matrix (fit done in u = k/s). Above it the
# 3×3 system is treated as singular/ill-conditioned -> status fit_failed.
COND_MAX = 1e8
# ATM sits near the smile's minimum, so the fitted a is often slightly below every observed IV. Allow
# that much slack on the [iv_min, iv_max] acceptance band before calling a fit fit_out_of_range.
FIT_RANGE_SLACK = 0.2

# Fully-qualified table names (Python stores names, never a second copy of the DDL).
IV_TERM_TABLE = "trading.iv_term_daily"
OPTION_TRADE_TABLE = "trading.option_trade"
LEDGER_TABLE = "trading.option_archive_ledger"

# Statuses, in the order they are checked (see fit.classify). Exactly one applies per group.
STATUS_TOO_FEW_TRADES = "too_few_trades"
STATUS_TOO_FEW_STRIKES = "too_few_strikes"
STATUS_ONE_SIDED = "one_sided"
STATUS_FIT_FAILED = "fit_failed"
STATUS_FIT_OUT_OF_RANGE = "fit_out_of_range"
STATUS_OK = "ok"
