"""
HAR realized-variance edge (HAR edge spec §3–§6).

- rv_daily.py  — daily realized variance from raw 1-minute bars (set-based SQL), §3.
- har.py       — the pooled HAR model: features, pooled least-squares fit, forecasts, §4/§5.
- __main__.py  — the backfill / verify / proof / forecast commands.
- ../tools/rv_eval.py — the walk-forward report, §6 (the gate; writes nothing).

Tables live in Scala's db/database_schema.sql (trading.rv_daily, trading.rv_forecast); schema.py
mirrors them so a fresh run can create them.
"""

MODEL_VERSION = "har-v1"
HORIZONS = (1, 5, 10, 22, 44, 63)  # trading days
WEEK_WINDOW = 5
MONTH_WINDOW = 22
MIN_SESSIONS = 66  # a name with fewer complete sessions has no edge and gets no forecast (§5)
TRADING_DAYS_PER_YEAR = 252
