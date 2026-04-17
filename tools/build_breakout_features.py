"""
tools/build_breakout_features.py

Build feature matrix for breakout candidate model from ClickHouse.
Joins breakout_candidate + breakout_outcome, adds pivot proximity features,
encodes categoricals, writes CSV + sidecar JSON.

Usage:
    PYTHONPATH=. python tools/build_breakout_features.py
    PYTHONPATH=. python tools/build_breakout_features.py \
        --start-date 2025-01-01 --end-date 2026-04-15 \
        --symbols NVDA,AMD
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.models.breakout.config import ModelConfig
from ml.shared.clickhouse import get_ch_client
from ml.shared.utils import utc_dt

log = logging.getLogger("BuildBreakoutFeatures")

DEFAULT_OUTPUT = "data/features/breakout/features.csv"

TIMEFRAME_ENCODING = {
    "1m": 0, "5m": 1, "10m": 2, "15m": 3, "20m": 4, "30m": 5, "60m": 6,
}

RIBBON_STATE_ENCODING = {
    "BULLISH_ALIGNED": 1,
    "ordered_bullish": 1,
    "BEARISH_ALIGNED": -1,
    "ordered_bearish": -1,
}


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def query_candidates_with_outcomes(
    ch,
    symbols: Optional[list[str]],
    start_date: Optional[date],
    end_date: Optional[date],
    timeframe: Optional[str] = None,
) -> list[dict]:
    """Fetch all candidates with labeled outcomes."""
    query = (
        "SELECT "
        "    bc.symbol, bc.timeframe, bc.ts, bc.session, bc.direction, "
        "    bc.price, bc.bar_index, bc.level_age_min, bc.ribbon_state, "
        "    bc.ribbon_age, bc.ribbon_spread, bc.atr, "
        "    bc.bar_range_atr, bc.bar_close_pct, bc.volume_ratio, "
        "    bc.gap_pct, bc.score, "
        "    bc.prior_session_high, bc.prior_session_low, "
        "    bo.max_favorable_excursion, bo.max_adverse_excursion "
        "FROM trading.breakout_candidate bc FINAL "
        "JOIN trading.breakout_outcome bo FINAL "
        "    ON bc.symbol = bo.symbol "
        "    AND bc.timeframe = bo.timeframe "
        "    AND bc.ts = bo.ts "
        "WHERE bo.entry_price IS NOT NULL "
    )
    params = {}

    if symbols:
        query += "AND bc.symbol IN %(symbols)s "
        params["symbols"] = symbols
    if start_date:
        query += "AND bc.session >= %(start)s "
        params["start"] = start_date
    if end_date:
        query += "AND bc.session <= %(end)s "
        params["end"] = end_date
    if timeframe:
        query += "AND bc.timeframe = %(tf)s "
        params["tf"] = timeframe

    query += "ORDER BY bc.ts"

    result = ch.query(query, parameters=params)

    cols = [
        "symbol", "timeframe", "ts", "session", "direction",
        "price", "bar_index", "level_age_min", "ribbon_state",
        "ribbon_age", "ribbon_spread", "atr",
        "bar_range_atr", "bar_close_pct", "volume_ratio",
        "gap_pct", "score",
        "prior_session_high", "prior_session_low",
        "mfe", "mae",
    ]
    rows = []
    for row in result.result_rows:
        d = dict(zip(cols, row))
        d["ts"] = utc_dt(d["ts"])
        d["session"] = d["session"] if isinstance(d["session"], date) else date.fromisoformat(str(d["session"]))
        d["mfe"] = float(d["mfe"]) if d["mfe"] is not None else None
        d["mae"] = float(d["mae"]) if d["mae"] is not None else None
        rows.append(d)
    return rows


def query_group_pivots(
    ch, symbol: str, timeframe: str, trading_date: date, max_ts: datetime,
) -> list[dict]:
    """Fetch recent pivots for a group, ordered by confirmed_ts DESC."""
    lookback_date = trading_date - timedelta(days=30)
    result = ch.query(
        "SELECT indicator, value, confirmed_ts, bar_index "
        "FROM trading.indicator FINAL "
        "WHERE symbol = %(symbol)s "
        "  AND timeframe = %(tf)s "
        "  AND indicator IN ('pivot_high', 'pivot_low') "
        "  AND confirmed_ts <= %(max_ts)s "
        "  AND trading_date >= %(lookback)s "
        "ORDER BY confirmed_ts DESC",
        parameters={
            "symbol": symbol,
            "tf": timeframe,
            "max_ts": max_ts,
            "lookback": lookback_date,
        },
    )
    return [
        {
            "indicator": row[0],
            "value": float(row[1]),
            "confirmed_ts": utc_dt(row[2]),
            "bar_index": int(row[3]),
        }
        for row in result.result_rows
    ]


def query_group_emas(
    ch, symbol: str, timeframe: str, trading_date: date,
) -> dict[datetime, dict[str, float]]:
    """Fetch all EMA ribbon values for a group, keyed by timestamp."""
    result = ch.query(
        "SELECT ts, indicator, value "
        "FROM trading.indicator FINAL "
        "WHERE symbol = %(symbol)s "
        "  AND timeframe = %(tf)s "
        "  AND trading_date = %(date)s "
        "  AND indicator IN ('ema10', 'ema15', 'ema20', 'ema25', 'ema30') "
        "  AND warm = true "
        "ORDER BY ts",
        parameters={
            "symbol": symbol,
            "tf": timeframe,
            "date": trading_date,
        },
    )
    ema_by_ts: dict[datetime, dict[str, float]] = {}
    for row in result.result_rows:
        ts = utc_dt(row[0])
        if ts not in ema_by_ts:
            ema_by_ts[ts] = {}
        ema_by_ts[ts][row[1]] = float(row[2])
    return ema_by_ts


# ---------------------------------------------------------------------------
# Pivot feature extraction
# ---------------------------------------------------------------------------

def extract_pivot_features(
    pivots: list[dict],
    candidate_ts: datetime,
    signal_price: float,
    atr: float,
    bar_index: int,
) -> dict:
    """Extract pivot proximity features for one candidate.

    pivots must be in confirmed_ts DESC order (from group query).
    Filters to confirmed_ts <= candidate_ts, then takes first 2 of each type.
    """
    highs = []
    lows = []
    for p in pivots:
        if p["confirmed_ts"] > candidate_ts:
            continue
        if p["indicator"] == "pivot_high" and len(highs) < 2:
            highs.append(p)
        elif p["indicator"] == "pivot_low" and len(lows) < 2:
            lows.append(p)
        if len(highs) >= 2 and len(lows) >= 2:
            break

    features = {}

    # Pivot high features
    if len(highs) >= 1 and atr > 0:
        features["pivot_high_1_dist_atr"] = (highs[0]["value"] - signal_price) / atr
        features["pivot_high_1_age_bars"] = bar_index - highs[0]["bar_index"]
    else:
        features["pivot_high_1_dist_atr"] = None
        features["pivot_high_1_age_bars"] = None

    if len(highs) >= 2 and atr > 0:
        features["pivot_high_2_dist_atr"] = (highs[1]["value"] - signal_price) / atr
        features["pivot_high_2_age_bars"] = bar_index - highs[1]["bar_index"]
    else:
        features["pivot_high_2_dist_atr"] = None
        features["pivot_high_2_age_bars"] = None

    # Pivot low features
    if len(lows) >= 1 and atr > 0:
        features["pivot_low_1_dist_atr"] = (signal_price - lows[0]["value"]) / atr
        features["pivot_low_1_age_bars"] = bar_index - lows[0]["bar_index"]
    else:
        features["pivot_low_1_dist_atr"] = None
        features["pivot_low_1_age_bars"] = None

    if len(lows) >= 2 and atr > 0:
        features["pivot_low_2_dist_atr"] = (signal_price - lows[1]["value"]) / atr
        features["pivot_low_2_age_bars"] = bar_index - lows[1]["bar_index"]
    else:
        features["pivot_low_2_dist_atr"] = None
        features["pivot_low_2_age_bars"] = None

    # Trend features
    if len(highs) >= 2:
        features["is_higher_high"] = 1 if highs[0]["value"] > highs[1]["value"] else 0
    else:
        features["is_higher_high"] = None

    if len(lows) >= 2:
        features["is_higher_low"] = 1 if lows[0]["value"] > lows[1]["value"] else 0
    else:
        features["is_higher_low"] = None

    return features


# ---------------------------------------------------------------------------
# Ribbon history feature extraction
# ---------------------------------------------------------------------------

def extract_ribbon_features(
    ema_by_ts: dict[datetime, dict[str, float]],
    candidate_ts: datetime,
    signal_price: float,
    bar_ts_list: list[datetime],
) -> dict:
    """Extract ribbon history features for one candidate."""
    null_result = {k: None for k in [
        "ribbon_spread_delta", "ribbon_spread_accel",
        "ema10_slope_3bar", "ema30_slope_3bar", "slope_differential",
        "was_compressed",
        "ema_ordered_count", "ribbon_upper_spread", "ribbon_lower_spread",
        "spread_ratio", "ema20_slope_3bar", "was_mixed_3bar", "is_resolving",
    ]}

    try:
        idx = bar_ts_list.index(candidate_ts)
    except ValueError:
        return null_result

    def get_emas(i) -> dict:
        if i < 0 or i >= len(bar_ts_list):
            return {}
        return ema_by_ts.get(bar_ts_list[i], {})

    now = get_emas(idx)
    ago1 = get_emas(idx - 1)
    ago3 = get_emas(idx - 3)

    ema10_now, ema30_now = now.get("ema10"), now.get("ema30")
    ema10_1, ema30_1 = ago1.get("ema10"), ago1.get("ema30")
    ema10_3, ema30_3 = ago3.get("ema10"), ago3.get("ema30")
    ema15_now = now.get("ema15")
    ema20_now = now.get("ema20")
    ema25_now = now.get("ema25")
    ema20_3 = ago3.get("ema20")

    if signal_price <= 0:
        return null_result

    spread_now = abs(ema10_now - ema30_now) / signal_price if ema10_now is not None and ema30_now is not None else None
    spread_1 = abs(ema10_1 - ema30_1) / signal_price if ema10_1 is not None and ema30_1 is not None else None
    spread_3 = abs(ema10_3 - ema30_3) / signal_price if ema10_3 is not None and ema30_3 is not None else None

    features = {}

    # Trajectory
    features["ribbon_spread_delta"] = (spread_now - spread_3) if spread_now is not None and spread_3 is not None else None
    features["ribbon_spread_accel"] = ((spread_now - spread_1) - (spread_1 - spread_3)) if spread_now is not None and spread_1 is not None and spread_3 is not None else None

    # Slopes
    features["ema10_slope_3bar"] = (ema10_now - ema10_3) / (3 * signal_price) if ema10_now is not None and ema10_3 is not None else None
    features["ema30_slope_3bar"] = (ema30_now - ema30_3) / (3 * signal_price) if ema30_now is not None and ema30_3 is not None else None
    features["slope_differential"] = (features["ema10_slope_3bar"] - features["ema30_slope_3bar"]) if features["ema10_slope_3bar"] is not None and features["ema30_slope_3bar"] is not None else None

    # Compression flag
    features["was_compressed"] = (1 if spread_3 < 0.001 else 0) if spread_3 is not None else None

    # --- Intermediate EMA features ---

    def ordered_count(d):
        vals = [d.get(k) for k in ("ema10", "ema15", "ema20", "ema25", "ema30")]
        if any(v is None for v in vals):
            return None
        pairs = list(zip(vals, vals[1:]))
        bull = sum(1 for a, b in pairs if a > b)
        bear = sum(1 for a, b in pairs if a < b)
        return bull if bull >= bear else -bear

    oc_now = ordered_count(now)
    oc_3 = ordered_count(ago3)
    features["ema_ordered_count"] = oc_now

    # Spread distribution
    if ema10_now is not None and ema20_now is not None:
        features["ribbon_upper_spread"] = (ema10_now - ema20_now) / signal_price
    else:
        features["ribbon_upper_spread"] = None

    if ema20_now is not None and ema30_now is not None:
        features["ribbon_lower_spread"] = (ema20_now - ema30_now) / signal_price
    else:
        features["ribbon_lower_spread"] = None

    if features["ribbon_upper_spread"] is not None and features["ribbon_lower_spread"] is not None and features["ribbon_lower_spread"] != 0:
        features["spread_ratio"] = features["ribbon_upper_spread"] / features["ribbon_lower_spread"]
    else:
        features["spread_ratio"] = None

    # Middle EMA slope
    if ema20_now is not None and ema20_3 is not None:
        features["ema20_slope_3bar"] = (ema20_now - ema20_3) / (3 * signal_price)
    else:
        features["ema20_slope_3bar"] = None

    # Twist progress
    was_mixed = oc_3 is not None and abs(oc_3) < 4
    features["was_mixed_3bar"] = (1 if was_mixed else 0) if oc_3 is not None else None
    features["is_resolving"] = (1 if was_mixed and oc_now is not None and abs(oc_now) >= 3 else 0) if oc_3 is not None and oc_now is not None else None

    return features


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def build_features(ch, candidates: list[dict],
                   mfe_threshold: float = 1.0,
                   mae_threshold: float = 1.0) -> pd.DataFrame:
    """Build full feature matrix with pivot features."""
    # Group by (symbol, timeframe, session) for batched pivot queries
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in candidates:
        groups[(c["symbol"], c["timeframe"], c["session"])].append(c)

    log.info("Computing pivot features for %d groups", len(groups))

    rows = []
    t0 = time.time()
    groups_done = 0

    for (symbol, tf, session_date), group in groups.items():
        # One pivot query per group
        max_ts = max(c["ts"] for c in group)
        pivots = query_group_pivots(ch, symbol, tf, session_date, max_ts)
        ema_by_ts = query_group_emas(ch, symbol, tf, session_date)
        bar_ts_list = sorted(ema_by_ts.keys())

        for c in group:
            pivot_feats = extract_pivot_features(
                pivots, c["ts"], float(c["price"]), float(c["atr"]), int(c["bar_index"])
            )
            ribbon_feats = extract_ribbon_features(
                ema_by_ts, c["ts"], float(c["price"]), bar_ts_list
            )

            row = {**c, **pivot_feats, **ribbon_feats}
            rows.append(row)

        groups_done += 1
        if groups_done % 200 == 0:
            elapsed = time.time() - t0
            log.info("Progress: %d/%d groups, %.0fs", groups_done, len(groups), elapsed)

    log.info("Pivot features complete: %d rows, %.1fs", len(rows), time.time() - t0)

    df = pd.DataFrame(rows)

    # --- Encode categoricals ---

    # Symbol: label encode alphabetically
    unique_symbols = sorted(df["symbol"].unique())
    symbol_map = {s: i for i, s in enumerate(unique_symbols)}
    df["symbol_encoded"] = df["symbol"].map(symbol_map)

    # Timeframe: fixed encoding
    df["timeframe_encoded"] = df["timeframe"].map(TIMEFRAME_ENCODING)

    # Direction
    df["direction_encoded"] = df["direction"].map({"long": 1, "short": -1})

    # Ribbon state
    df["ribbon_state_encoded"] = df["ribbon_state"].map(
        lambda s: RIBBON_STATE_ENCODING.get(s, 0)
    )

    # --- Prior session distance features (ATR-normalized) ---
    df["prior_session_high_dist_atr"] = (df["prior_session_high"] - df["price"]) / df["atr"]
    df["prior_session_low_dist_atr"] = (df["price"] - df["prior_session_low"]) / df["atr"]

    # --- Label ---
    df["label"] = ((df["mfe"] >= mfe_threshold) & (df["mae"] <= mae_threshold)).astype(int)

    # --- Select and order output columns ---
    output_cols = [
        "ts", "symbol", "symbol_encoded", "timeframe", "timeframe_encoded",
        "direction_encoded",
        "bar_index", "level_age_min", "ribbon_state_encoded", "ribbon_age",
        "ribbon_spread", "atr", "bar_range_atr", "bar_close_pct",
        "volume_ratio", "gap_pct", "score",
        "pivot_high_1_dist_atr", "pivot_high_1_age_bars",
        "pivot_high_2_dist_atr", "pivot_high_2_age_bars",
        "pivot_low_1_dist_atr", "pivot_low_1_age_bars",
        "pivot_low_2_dist_atr", "pivot_low_2_age_bars",
        "is_higher_high", "is_higher_low",
        "prior_session_high_dist_atr", "prior_session_low_dist_atr",
        "ribbon_spread_delta", "ribbon_spread_accel",
        "ema10_slope_3bar", "ema30_slope_3bar", "slope_differential",
        "was_compressed",
        "ema_ordered_count", "ribbon_upper_spread", "ribbon_lower_spread",
        "spread_ratio", "ema20_slope_3bar", "was_mixed_3bar", "is_resolving",
        "mfe", "mae", "label",
    ]
    df = df[output_cols]

    return df, symbol_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build feature matrix for breakout candidate model",
    )
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config YAML")
    parser.add_argument("--timeframe", type=str, default=None,
                        help="Filter to single timeframe (e.g. 5m)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config if provided — CLI flags override config values
    cfg = ModelConfig.from_yaml(args.config) if args.config else None
    timeframe = args.timeframe or (cfg.timeframe if cfg else None)
    mfe_threshold = cfg.mfe_threshold if cfg else 1.0
    mae_threshold = cfg.mae_threshold if cfg else 1.0

    start = date.fromisoformat(args.start_date) if args.start_date else None
    end = date.fromisoformat(args.end_date) if args.end_date else None
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None

    ch = get_ch_client()

    log.info("Querying candidates with outcomes...")
    candidates = query_candidates_with_outcomes(ch, symbols, start, end, timeframe)
    log.info("Loaded %d labeled candidates", len(candidates))

    if not candidates:
        log.info("No candidates found")
        return

    df, symbol_map = build_features(ch, candidates, mfe_threshold, mae_threshold)

    # Write output
    if args.output != DEFAULT_OUTPUT:
        output_path = Path(args.output)
    elif timeframe:
        output_path = Path(f"data/features/breakout/features_{timeframe}.csv")
    else:
        output_path = Path(DEFAULT_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    log.info("Wrote %d rows to %s", len(df), output_path)

    # Write sidecar encoding
    enc_name = f"encodings_{timeframe}.json" if timeframe else "encodings.json"
    encodings_path = output_path.parent / enc_name
    encodings = {
        "symbol": symbol_map,
        "timeframe": TIMEFRAME_ENCODING,
        "direction": {"long": 1, "short": -1},
        "ribbon_state": RIBBON_STATE_ENCODING,
    }
    with open(encodings_path, "w") as f:
        json.dump(encodings, f, indent=2)
    log.info("Wrote encodings to %s", encodings_path)

    # Summary
    pos = (df["label"] == 1).sum()
    neg = (df["label"] == 0).sum()
    log.info("Label distribution (mfe>=%s AND mae<=%s): %d positive (%.1f%%), %d negative (%.1f%%)",
             mfe_threshold, mae_threshold, pos, 100 * pos / len(df), neg, 100 * neg / len(df))


if __name__ == "__main__":
    main()
