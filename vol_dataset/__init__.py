"""
The vol model's training dataset (roadmap-2026-11-16 Stage 4): versioned Parquet, one input builder
shared by training and the later live forecast.

One row per (underlying, t, H): t a trading day, H in HORIZONS_TD (the iv_curve horizons). A row exists
when iv_curve_daily (iso_v1) has an ok horizon row for (underlying, t, H) and rv_daily has a row for
each of the H trading days after t; rows with a corporate action / split in (t, t+H] or a zero
realized semivariance sum are dropped and counted. Targets are the log semivariance ratios
(pc-objective-function-draft.md §7); inputs use only data available at t's close.

- rules.py   — the pure math (C/J split, the earnings-date rule, reaction session, event-window sums,
               targets), no ClickHouse, unit-tested (mirrors iv_event/fwd.py).
- inputs.py  — the ONE builder: ClickHouse reads -> DataFrames -> a pure assemble_inputs(frames,
               month) -> per-(underlying, t, H) inputs. build_inputs wraps the reads around it; the
               live forecast will call build_inputs for a single day. Look-ahead-safe: inputs for t
               never depend on a source row dated after t (the one documented exception is the
               confirmed earnings date, which the vendor publishes ahead — see assemble_inputs).
- targets.py — the forward targets and the drop rules (NOT part of the look-ahead-safe input path).
- dataset.py — the `build` driver: month by month, join inputs to targets, write one Parquet per year
               (temp file + atomic rename) and manifest.json.
- summary.py — the read-only `summary` over the written files.

This package reads ClickHouse and writes Parquet under DATASET_DIR; it creates no ClickHouse tables.
"""

from __future__ import annotations

from pathlib import Path

from ml.shared.paths import DATA_DIR

# Reused verbatim from iv_event so the detector's event rule and the market-wide rule never drift.
from iv_event import (  # noqa: F401  (re-exported for callers/tests)
    EVENT_MAX_EFF_DAYS, EVENT_MAX_START_TD, EVENT_MIN_EXCESS_DAYS, MARKET_WIDE_MIN_NAMES,
)

# ── version & output location ───────────────────────────────────────────────────────────────────────
DATASET_VERSION = "v1"
# Persistent data root (ml/shared/paths.DATA_DIR = repo data/, /opt/marketdata-python/app/data on
# drogon — excluded from the deploy pack, so it survives deploys). One Parquet per year lives here.
DATASET_DIR = DATA_DIR / "vol_dataset" / DATASET_VERSION
MANIFEST_NAME = "manifest.json"

# ── the row grid ─────────────────────────────────────────────────────────────────────────────────────
HORIZONS_TD = (5, 10, 21, 42)   # the iv_curve_daily horizon_td values a row is built for

# ── input window lengths (all named here; nothing defaulted elsewhere) ────────────────────────────────
MEAN_WINDOWS = (1, 5, 22)       # trailing-session means of C, J, rs_down, rs_up
UNSPLIT_WINDOW = 22             # n_unsplit_days_22: NULL-intraday-flag days in the last this many
SURPRISE_WINDOW = 252           # trailing sessions the surprise (unexplained-jump) stats cover
REGIME_WINDOW = 22              # SPY C/J means and the market-wide count cover the last this many sessions

# ── the earnings-date rule (as known at t; look-ahead-safe except the confirmed date, published ahead) ─
EARNINGS_CONFIRMED_DAYS = 21    # a report with reported_date within this many calendar days ahead of t is confirmed
EARNINGS_YEAR_DAYS = 364        # expected date = a past report's reported_date + this (52 weeks, weekday-stable)
EARNINGS_HISTORY_N = 8          # own-earnings outcomes (end_expiry < t) the history medians cover

# Read lookback: enough calendar days to cover SURPRISE_WINDOW trading sessions (252 * 7/5 ≈ 353) with
# a holiday buffer. Earnings history and own-earnings outcomes are read per-name unbounded (small).
LOOKBACK_CALENDAR_DAYS = 420
# Forward read for the targets: enough calendar days to cover the longest horizon (42 trading days
# ≈ 59 calendar) after the last t in the month, with a buffer.
FORWARD_CALENDAR_DAYS = 90

SPY_SYMBOL = "SPY"

# Vendor report-time vocabulary (earnings_history.vendor_report_time) and our status/time labels.
REPORT_TIME_PRE = "pre-market"
REPORT_TIME_POST = "post-market"
REPORT_TIME_UNKNOWN = "unknown"
EARNINGS_STATUS_CONFIRMED = "confirmed"
EARNINGS_STATUS_EXPECTED = "expected"
EARNINGS_STATUS_UNKNOWN = "unknown"

# Fully-qualified table names (stored here, never a second copy of any DDL).
RV_DAILY_TABLE = "trading.rv_daily"
JUMP_DAILY_TABLE = "trading.jump_daily"
IV_CURVE_TABLE = "trading.iv_curve_daily"
IV_EVENT_TABLE = "trading.iv_event_window"
IV_EVENT_OUTCOME_TABLE = "trading.iv_event_outcome"
EARNINGS_HISTORY_TABLE = "trading.earnings_history"

# Source method / status tags (match the upstream packages).
RV_SOURCE = "har-rv"
JUMP_METHOD = "bns_v1"
CURVE_METHOD = "iso_v1"
CURVE_KIND_HORIZON = "horizon"
CURVE_STATUS_OK = "ok"
WINDOW_METHOD = "fwd_v1"
WINDOW_STATUS_OK = "ok"
OUTCOME_METHOD = "out_v1"
OUTCOME_OWN_EARNINGS = "own_earnings"

assert isinstance(DATASET_DIR, Path)
