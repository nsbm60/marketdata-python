"""
tools/label_outcomes.py

Compute outcome labels for breakout candidates and write to
trading.breakout_outcome.  Joins back to breakout_candidate by
(symbol, timeframe, ts) at query time.

Three exit strategies per candidate:
  - pivot:  first confirmed opposing pivot after entry (EOD fallback)
  - ribbon: first bar where price re-enters EMA ribbon (EOD fallback)
  - eod:    session close (always computed)

MFE/MAE measured from entry to EOD regardless of exit strategy.

Usage:
    python tools/label_outcomes.py \
        --start-date 2025-01-01 \
        --end-date 2026-04-15 \
        --symbols NVDA,AMD \
        --force \
        --flush-size 500
"""

import argparse
import logging
import time
from collections import defaultdict
from datetime import date, datetime
from dataclasses import dataclass
from typing import Optional

from discovery.service_locator import ServiceLocator
from ml.models.breakout.config import ModelConfig
from ml.shared.clickhouse import get_ch_client
from ml.shared.config import fetch_symbol_list
from ml.shared.utils import utc_dt

log = logging.getLogger("LabelOutcomes")

OUTCOME_TABLE = "trading.breakout_outcome"
OUTCOME_COLUMNS = [
    "symbol", "timeframe", "ts",
    "entry_price",
    "return_atr_pivot", "exit_bars_pivot",
    "return_atr_ribbon", "exit_bars_ribbon",
    "return_atr_ema30", "exit_bars_ema30",
    "return_atr_eod",
    "max_favorable_excursion", "max_adverse_excursion",
    "return_atr_target", "target_hit", "target_bars_to_hit",
    "bracket_exit_type", "bracket_return_atr", "bracket_bars_to_exit",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    symbol: str
    timeframe: str
    ts: datetime
    session: date
    direction: str      # "long" or "short"
    atr: float


@dataclass
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class Pivot:
    indicator: str      # "pivot_high" or "pivot_low"
    value: float
    confirmed_ts: datetime


# ---------------------------------------------------------------------------
# ClickHouse queries
# ---------------------------------------------------------------------------

def query_candidates(
    ch,
    symbols: Optional[list[str]],
    start_date: Optional[date],
    end_date: Optional[date],
    force: bool,
) -> list[Candidate]:
    """Load candidates that need outcome labeling."""
    if force:
        query = (
            "SELECT symbol, timeframe, ts, session, direction, atr "
            "FROM trading.breakout_candidate FINAL "
            "WHERE 1=1 "
        )
    else:
        query = (
            "SELECT bc.symbol, bc.timeframe, bc.ts, bc.session, "
            "       bc.direction, bc.atr "
            "FROM trading.breakout_candidate bc FINAL "
            "LEFT JOIN trading.breakout_outcome bo FINAL "
            "  ON bc.symbol = bo.symbol "
            "  AND bc.timeframe = bo.timeframe "
            "  AND bc.ts = bo.ts "
            "WHERE bo.ts IS NULL "
        )

    params = {}

    if symbols:
        if force:
            query += "AND symbol IN %(symbols)s "
        else:
            query += "AND bc.symbol IN %(symbols)s "
        params["symbols"] = symbols

    if start_date:
        col = "session" if force else "bc.session"
        query += f"AND {col} >= %(start)s "
        params["start"] = start_date

    if end_date:
        col = "session" if force else "bc.session"
        query += f"AND {col} <= %(end)s "
        params["end"] = end_date

    col_prefix = "" if force else "bc."
    query += f"ORDER BY {col_prefix}symbol, {col_prefix}session, {col_prefix}timeframe, {col_prefix}ts"

    result = ch.query(query, parameters=params)

    candidates = []
    for row in result.result_rows:
        candidates.append(Candidate(
            symbol=row[0],
            timeframe=row[1],
            ts=utc_dt(row[2]),
            session=row[3] if isinstance(row[3], date) else date.fromisoformat(str(row[3])),
            direction=row[4],
            atr=float(row[5]),
        ))
    return candidates


def query_session_bars(
    ch, symbol: str, timeframe: str, trading_date: date,
) -> list[Bar]:
    """Fetch regular-session bars for one symbol/tf/date."""
    result = ch.query(
        "SELECT ts, open, high, low, close "
        "FROM trading.stock_bar FINAL "
        "WHERE symbol = %(symbol)s "
        "  AND period = %(tf)s "
        "  AND session = 'regular' "
        "  AND trading_date = %(date)s "
        "ORDER BY ts",
        parameters={
            "symbol": symbol,
            "tf": timeframe,
            "date": trading_date,
        },
    )
    return [
        Bar(
            ts=utc_dt(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
        )
        for row in result.result_rows
    ]


def query_session_pivots(
    ch, symbol: str, timeframe: str, trading_date: date,
) -> list[Pivot]:
    """Fetch confirmed pivots for one symbol/tf/date, sorted by confirmed_ts."""
    result = ch.query(
        "SELECT indicator, value, confirmed_ts "
        "FROM trading.indicator FINAL "
        "WHERE symbol = %(symbol)s "
        "  AND timeframe = %(tf)s "
        "  AND trading_date = %(date)s "
        "  AND indicator IN ('pivot_high', 'pivot_low') "
        "ORDER BY confirmed_ts",
        parameters={
            "symbol": symbol,
            "tf": timeframe,
            "date": trading_date,
        },
    )
    return [
        Pivot(
            indicator=row[0],
            value=float(row[1]),
            confirmed_ts=utc_dt(row[2]),
        )
        for row in result.result_rows
    ]


def query_session_emas(
    ch, symbol: str, timeframe: str, trading_date: date,
) -> tuple[dict[datetime, float], dict[datetime, float]]:
    """Fetch warm EMA10 and EMA30 values keyed by bar timestamp."""
    result = ch.query(
        "SELECT ts, indicator, value "
        "FROM trading.indicator FINAL "
        "WHERE symbol = %(symbol)s "
        "  AND timeframe = %(tf)s "
        "  AND trading_date = %(date)s "
        "  AND indicator IN ('ema10', 'ema30') "
        "  AND warm = true "
        "ORDER BY ts",
        parameters={
            "symbol": symbol,
            "tf": timeframe,
            "date": trading_date,
        },
    )
    ema10: dict[datetime, float] = {}
    ema30: dict[datetime, float] = {}
    for row in result.result_rows:
        ts = utc_dt(row[0])
        if row[1] == "ema10":
            ema10[ts] = float(row[2])
        elif row[1] == "ema30":
            ema30[ts] = float(row[2])
    return ema10, ema30


# ---------------------------------------------------------------------------
# Labeling logic
# ---------------------------------------------------------------------------

def compute_target_exit(
    bars: list[Bar],
    entry_price: float,
    direction: str,
    atr: float,
    target_atr: float,
) -> tuple[float, bool, Optional[int]]:
    """Walk forward to find first bar where ATR target is hit.

    Returns (return_atr, target_hit, bars_to_hit).
    """
    target_price = (entry_price + target_atr * atr
                    if direction == "long"
                    else entry_price - target_atr * atr)

    for i, bar in enumerate(bars[1:]):  # skip entry bar
        if direction == "long" and bar.high >= target_price:
            return target_atr, True, i + 2
        elif direction == "short" and bar.low <= target_price:
            return target_atr, True, i + 2

    if bars:
        last_bar = bars[-1]
        eod_return = ((last_bar.close - entry_price) / atr
                      if direction == "long"
                      else (entry_price - last_bar.close) / atr)
        return eod_return, False, None

    return 0.0, False, None


def compute_bracket_exit(
    bars: list[Bar],
    entry_price: float,
    direction: str,
    atr: float,
    target_atr: float,
    stop_atr: float,
) -> tuple[str, float, Optional[int]]:
    """Simulate bracket order: target limit + stop loss."""
    if direction == "long":
        target_price = entry_price + target_atr * atr
        stop_price = entry_price - stop_atr * atr
    else:
        target_price = entry_price - target_atr * atr
        stop_price = entry_price + stop_atr * atr

    for i, bar in enumerate(bars[1:]):  # skip entry bar
        if direction == "long":
            target_hit = bar.high >= target_price
            stop_hit = bar.low <= stop_price
        else:
            target_hit = bar.low <= target_price
            stop_hit = bar.high >= stop_price

        if target_hit and stop_hit:
            return "stop", -stop_atr, i + 2
        elif target_hit:
            return "target", target_atr, i + 2
        elif stop_hit:
            return "stop", -stop_atr, i + 2

    if bars:
        last_bar = bars[-1]
        eod_return = ((last_bar.close - entry_price) / atr
                      if direction == "long"
                      else (entry_price - last_bar.close) / atr)
        return "eod", eod_return, None

    return "eod", 0.0, None


def label_group(
    candidates: list[Candidate],
    bars: list[Bar],
    pivots: list[Pivot],
    ema10_by_ts: dict[datetime, float],
    ema30_by_ts: dict[datetime, float],
    target_atr: float = 1.0,
    stop_atr: float = 0.5,
) -> list[list]:
    """Compute outcome labels for a group of candidates sharing (symbol, tf, date).

    Returns list of rows matching OUTCOME_COLUMNS ordering.
    """
    rows = []

    for cand in candidates:
        direction = 1 if cand.direction == "long" else -1

        # Find entry bar: first bar with ts > candidate.ts
        entry_idx = None
        for i, bar in enumerate(bars):
            if bar.ts > cand.ts:
                entry_idx = i
                break

        if entry_idx is None:
            # No next bar — write row with all-null outcomes
            rows.append([
                cand.symbol, cand.timeframe, cand.ts,
                None, None, None, None, None, None, None, None, None, None,
                None, 0, None,
                "", None, None,
            ])
            continue

        entry_price = bars[entry_idx].open
        entry_ts = bars[entry_idx].ts
        last_bar = bars[-1]
        range_bars = bars[entry_idx:]

        # --- EOD exit (always computed) ---
        return_atr_eod = direction * (last_bar.close - entry_price) / cand.atr

        # --- MFE / MAE (entry to EOD) ---
        if direction == 1:
            # Long: favorable = highs, adverse = lows
            max_high = max(b.high for b in range_bars)
            min_low = min(b.low for b in range_bars)
            mfe = (max_high - entry_price) / cand.atr
            mae = (entry_price - min_low) / cand.atr
        else:
            # Short: favorable = lows, adverse = highs
            min_low = min(b.low for b in range_bars)
            max_high = max(b.high for b in range_bars)
            mfe = (entry_price - min_low) / cand.atr
            mae = (max_high - entry_price) / cand.atr

        # --- Pivot exit ---
        opposing = "pivot_low" if direction == 1 else "pivot_high"
        pivot_exit_bar = None
        for p in pivots:
            if p.confirmed_ts > entry_ts and p.indicator == opposing:
                # Find bar at confirmed_ts
                for b in range_bars:
                    if b.ts == p.confirmed_ts:
                        pivot_exit_bar = b
                        break
                if pivot_exit_bar is not None:
                    break

        if pivot_exit_bar is not None:
            return_atr_pivot = direction * (pivot_exit_bar.close - entry_price) / cand.atr
            exit_bars_pivot = _count_bars(range_bars, entry_ts, pivot_exit_bar.ts)
        else:
            # EOD fallback
            return_atr_pivot = return_atr_eod
            exit_bars_pivot = len(range_bars)

        # --- Ribbon exit ---
        ribbon_exit_bar = None
        for b in range_bars[1:]:  # skip entry bar itself
            ema10 = ema10_by_ts.get(b.ts)
            if ema10 is None:
                continue
            if direction == 1 and b.close < ema10:
                ribbon_exit_bar = b
                break
            elif direction == -1 and b.close > ema10:
                ribbon_exit_bar = b
                break

        if ribbon_exit_bar is not None:
            return_atr_ribbon = direction * (ribbon_exit_bar.close - entry_price) / cand.atr
            exit_bars_ribbon = _count_bars(range_bars, entry_ts, ribbon_exit_bar.ts)
        else:
            # EOD fallback
            return_atr_ribbon = return_atr_eod
            exit_bars_ribbon = len(range_bars)

        # --- EMA30 exit ---
        ema30_exit_bar = None
        for b in range_bars[1:]:  # skip entry bar itself
            ema30 = ema30_by_ts.get(b.ts)
            if ema30 is None:
                continue
            if direction == 1 and b.close < ema30:
                ema30_exit_bar = b
                break
            elif direction == -1 and b.close > ema30:
                ema30_exit_bar = b
                break

        if ema30_exit_bar is not None:
            return_atr_ema30 = direction * (ema30_exit_bar.close - entry_price) / cand.atr
            exit_bars_ema30 = _count_bars(range_bars, entry_ts, ema30_exit_bar.ts)
        else:
            # EOD fallback
            return_atr_ema30 = return_atr_eod
            exit_bars_ema30 = len(range_bars)

        # --- Target exit ---
        return_atr_target, target_hit, target_bars_to_hit = compute_target_exit(
            range_bars, entry_price, cand.direction, cand.atr, target_atr
        )

        # --- Bracket exit ---
        bracket_exit_type, bracket_return_atr, bracket_bars_to_exit = compute_bracket_exit(
            range_bars, entry_price, cand.direction, cand.atr, target_atr, stop_atr
        )

        rows.append([
            cand.symbol,
            cand.timeframe,
            cand.ts,
            entry_price,
            return_atr_pivot,
            exit_bars_pivot,
            return_atr_ribbon,
            exit_bars_ribbon,
            return_atr_ema30,
            exit_bars_ema30,
            return_atr_eod,
            mfe,
            mae,
            return_atr_target,
            1 if target_hit else 0,
            target_bars_to_hit,
            bracket_exit_type,
            bracket_return_atr,
            bracket_bars_to_exit,
        ])

    return rows


def _count_bars(bars: list[Bar], entry_ts: datetime, exit_ts: datetime) -> int:
    """Count bars from entry to exit inclusive."""
    count = 0
    for b in bars:
        if b.ts >= entry_ts and b.ts <= exit_ts:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def flush_outcomes(ch, buffer: list[list]) -> None:
    """Insert buffered outcome rows to ClickHouse."""
    if not buffer:
        return
    count = len(buffer)
    try:
        ch.insert(OUTCOME_TABLE, buffer, column_names=OUTCOME_COLUMNS)
        log.info("Flushed %d outcome rows", count)
    except Exception as e:
        log.error("Failed to flush %d outcome rows: %s", count, e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Label breakout candidates with outcome metrics",
    )
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date (YYYY-MM-DD), filters on session/trading_date")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (default: trading_universe from MDS)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config YAML (for target_atr)")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess all rows, not just unlabeled ones")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute outcomes but do not write to ClickHouse")
    parser.add_argument("--flush-size", type=int, default=500,
                        help="ClickHouse insert batch size (default: 500)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = ModelConfig.from_yaml(args.config) if args.config else None
    target_atr = cfg.target_atr if cfg else 1.0
    stop_atr = cfg.stop_atr if cfg else 0.5

    start = date.fromisoformat(args.start_date) if args.start_date else None
    end = date.fromisoformat(args.end_date) if args.end_date else None

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        md = ServiceLocator.wait_for_service(ServiceLocator.MARKET_DATA, timeout_sec=30)
        log.info("MDS found: %s", md.router)
        symbols = fetch_symbol_list(md.router, "trading_universe")
        log.info("Using TradingUniverse: %d symbols", len(symbols))

    ch = get_ch_client()

    log.info(
        "Loading candidates (force=%s, symbols=%s, start=%s, end=%s)",
        args.force, symbols, start, end,
    )
    candidates = query_candidates(ch, symbols, start, end, args.force)
    log.info("Loaded %d candidates to label", len(candidates))

    # Filter to config's timeframe when --config is provided
    if cfg:
        candidates = [c for c in candidates if c.timeframe == cfg.timeframe]
        log.info("Filtered to %s: %d candidates", cfg.timeframe, len(candidates))

    if not candidates:
        log.info("Nothing to do")
        return

    # Group by (symbol, timeframe, session)
    groups: dict[tuple, list[Candidate]] = defaultdict(list)
    for c in candidates:
        groups[(c.symbol, c.timeframe, c.session)].append(c)

    log.info("Processing %d groups", len(groups))

    buffer: list[list] = []
    total_labeled = 0
    total_skipped = 0
    t0 = time.time()

    for (symbol, tf, session_date), group_candidates in groups.items():
        try:
            bars = query_session_bars(ch, symbol, tf, session_date)
            if not bars:
                log.debug("%s %s %s: no bars, skipping %d candidates",
                          symbol, tf, session_date, len(group_candidates))
                # Write null rows for all candidates in this group
                for c in group_candidates:
                    buffer.append([
                        c.symbol, c.timeframe, c.ts,
                        None, None, None, None, None, None, None, None, None, None,
                        None, 0, None,
                        "", None, None,
                    ])
                    total_skipped += 1
                continue

            pivots = query_session_pivots(ch, symbol, tf, session_date)
            ema10, ema30 = query_session_emas(ch, symbol, tf, session_date)

            rows = label_group(group_candidates, bars, pivots, ema10, ema30, target_atr, stop_atr)
            buffer.extend(rows)

            labeled = sum(1 for r in rows if r[3] is not None)  # entry_price not null
            skipped = len(rows) - labeled
            total_labeled += labeled
            total_skipped += skipped

            log.debug("%s %s %s: %d labeled, %d skipped (no entry bar)",
                      symbol, tf, session_date, labeled, skipped)

        except Exception as e:
            log.error("%s %s %s: %s", symbol, tf, session_date, e)
            continue

        if len(buffer) >= args.flush_size and not args.dry_run:
            flush_outcomes(ch, buffer)
            buffer.clear()

    if buffer and not args.dry_run:
        flush_outcomes(ch, buffer)
        buffer.clear()

    elapsed = time.time() - t0
    log.info(
        "Complete: %d labeled, %d skipped, %.1fs elapsed",
        total_labeled, total_skipped, elapsed,
    )


if __name__ == "__main__":
    main()
