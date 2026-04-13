"""
tools/replay_breakouts.py

Historical replay of breakout detection across all timeframes and symbols.
Reads bars and indicators from ClickHouse, runs them through BreakoutEngine,
and persists candidates to trading.breakout_candidate with source='replay'.

Requires MDS running (for get_trading_sessions RPC and symbol list).

Usage:
    python tools/replay_breakouts.py \
        --start-date 2020-12-31 \
        --end-date 2026-04-11 \
        --symbols NVDA,AMD \
        --timeframes 5m,15m \
        --flush-size 200

Before first run, verify that stock_bar.ts and indicator.ts use identical
timestamps for the same bar (spot check a recent date).
"""

import argparse
import logging
import time
from collections import defaultdict
from datetime import date, datetime
from typing import Optional

from discovery.service_locator import ServiceLocator
from ml.models.breakout.engine import BreakoutEngine
from ml.models.breakout.persistence import BreakoutPersistor
from ml.models.breakout.signal_generator import SignalConfig
from ml.models.breakout.types import Bar
from ml.shared.clickhouse import get_ch_client
from ml.shared.config import fetch_symbol_list
from ml.shared.mds_client import TradingSession, get_trading_sessions

log = logging.getLogger("ReplayBreakouts")

DEFAULT_TIMEFRAMES = ["1m", "5m", "10m", "15m", "20m", "30m", "60m"]


# ---------------------------------------------------------------------------
# ClickHouse queries — all use UTC timestamp ranges, no timezone conversion
# ---------------------------------------------------------------------------

def query_prior_session(
    ch, symbols: list[str], session: TradingSession, period: str,
) -> dict[str, tuple[float, float, float]]:
    """Fetch prior session close/high/low per symbol.

    Uses the session's UTC open/close range directly.
    Returns {symbol: (close, high, low)}.
    """
    if not symbols:
        return {}
    result = ch.query(
        "SELECT symbol, "
        "  argMax(close, ts) AS prior_close, "
        "  max(high) AS prior_high, "
        "  min(low) AS prior_low "
        "FROM trading.stock_bar FINAL "
        "WHERE symbol IN %(symbols)s "
        "  AND period = %(period)s "
        "  AND session = 'regular' "
        "  AND ts >= %(open_utc)s AND ts < %(close_utc)s "
        "GROUP BY symbol",
        parameters={
            "symbols": symbols,
            "period": period,
            "open_utc": session.open_utc,
            "close_utc": session.close_utc,
        },
    )
    return {row[0]: (row[1], row[2], row[3]) for row in result.result_rows}


def query_bars(
    ch, symbol: str, session: TradingSession, period: str,
) -> list[Bar]:
    """Fetch bars for one symbol within a session's UTC range."""
    result = ch.query(
        "SELECT ts, open, high, low, close, volume, vwap "
        "FROM trading.stock_bar FINAL "
        "WHERE symbol = %(symbol)s "
        "  AND period = %(period)s "
        "  AND session = 'regular' "
        "  AND ts >= %(open_utc)s AND ts < %(close_utc)s "
        "ORDER BY ts",
        parameters={
            "symbol": symbol,
            "period": period,
            "open_utc": session.open_utc,
            "close_utc": session.close_utc,
        },
    )
    bars = []
    for row in result.result_rows:
        bars.append(Bar(
            ts=row[0],
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=int(row[5]),
            vwap=float(row[6]) if row[6] else float(row[4]),
        ))
    return bars


def query_indicators(
    ch, symbol: str, session: TradingSession, timeframe: str,
) -> list[tuple[datetime, str, float]]:
    """Fetch warm indicator rows for one symbol within a session's UTC range.

    Returns list of (ts, indicator_name, value).
    """
    result = ch.query(
        "SELECT ts, indicator, value "
        "FROM trading.indicator FINAL "
        "WHERE symbol = %(symbol)s "
        "  AND timeframe = %(tf)s "
        "  AND warm = true "
        "  AND ts >= %(open_utc)s AND ts < %(close_utc)s "
        "ORDER BY ts, indicator",
        parameters={
            "symbol": symbol,
            "tf": timeframe,
            "open_utc": session.open_utc,
            "close_utc": session.close_utc,
        },
    )
    return [(row[0], row[1], float(row[2])) for row in result.result_rows]


def _derive_ribbon_state(ema_dict: dict[str, float]) -> str:
    """Derive ribbon state from EMA values.

    BULLISH_ALIGNED: ema10 > ema15 > ema20 > ema25 > ema30
    BEARISH_ALIGNED: ema10 < ema15 < ema20 < ema25 < ema30
    MIXED: anything else
    """
    keys = ["ema10", "ema15", "ema20", "ema25", "ema30"]
    vals = [ema_dict.get(k) for k in keys]
    if any(v is None for v in vals):
        return "MIXED"

    if all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
        return "BULLISH_ALIGNED"
    if all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
        return "BEARISH_ALIGNED"
    return "MIXED"


def pivot_indicators(
    rows: list[tuple[datetime, str, float]],
) -> dict[datetime, dict]:
    """Pivot EAV indicator rows into per-timestamp dicts.

    The indicator table stores EMA values (ema10..ema30) and ATR.
    Ribbon state is derived from EMA ordering since it is not stored
    in the indicator table.

    Returns {ts: {kwargs for engine.update_indicator()}}.
    """
    by_ts: dict[datetime, dict[str, float]] = defaultdict(dict)
    for ts, indicator, value in rows:
        by_ts[ts][indicator] = value

    result = {}
    for ts, indicators in by_ts.items():
        ema_dict = {}
        atr_value = None
        ema_warm = False
        atr_warm = False

        for name, val in indicators.items():
            if name.startswith("ema"):
                ema_dict[name] = val
            elif name == "atr":
                atr_value = val
                atr_warm = True

        if ema_dict:
            ema_warm = True

        result[ts] = {
            "ema_dict": ema_dict if ema_dict else None,
            "atr_value": atr_value,
            "ribbon_state": _derive_ribbon_state(ema_dict) if ema_dict else None,
            "ema_warm": ema_warm,
            "atr_warm": atr_warm,
        }

    return result


# ---------------------------------------------------------------------------
# Replay loop
# ---------------------------------------------------------------------------

def replay(
    ch,
    symbols: list[str],
    timeframes: list[str],
    sessions: list[TradingSession],
    flush_size: int = 200,
) -> int:
    """Run historical replay. Returns total candidate count."""

    log.info(
        "Replay: %d trading days, %d symbols, %d timeframes",
        len(sessions), len(symbols), len(timeframes),
    )

    if not sessions:
        log.warning("No trading sessions in range")
        return 0

    persistor = BreakoutPersistor(ch, flush_size=flush_size)

    # Loose thresholds for replay — cast a wide net, let Model 2 learn
    signal_config = SignalConfig()

    total_candidates = 0
    t0 = time.time()

    for day_idx, session in enumerate(sessions):
        prior_session = sessions[day_idx - 1] if day_idx > 0 else None
        day_candidates = 0

        for tf in timeframes:
            engine = BreakoutEngine(tf, signal_config)
            engine.init_symbols(symbols)

            # Seed prior session
            if prior_session is not None:
                prior = query_prior_session(ch, symbols, prior_session, tf)
                for symbol, (close, high, low) in prior.items():
                    engine.set_prior_session(symbol, close, high, low)

            # Process each symbol
            tf_candidates = 0
            tf_symbols_with_candidates = set()

            for symbol in symbols:
                try:
                    bars = query_bars(ch, symbol, session, tf)
                    if not bars:
                        continue

                    indicators = query_indicators(ch, symbol, session, tf)
                    ind_by_ts = pivot_indicators(indicators)

                    for bar in bars:
                        if bar.ts in ind_by_ts:
                            engine.update_indicator(symbol, **ind_by_ts[bar.ts])
                        candidates = engine.on_bar(symbol, bar)
                        for c in candidates:
                            persistor.persist(c, tf, session.date, source="replay")
                            tf_candidates += 1
                            tf_symbols_with_candidates.add(symbol)

                except Exception as e:
                    log.warning("%s %s %s: %s", session.date, tf, symbol, e)
                    continue

            if tf_candidates > 0:
                log.info(
                    "%s %s: %d candidates across %d symbols",
                    session.date, tf, tf_candidates,
                    len(tf_symbols_with_candidates),
                )

            day_candidates += tf_candidates

        total_candidates += day_candidates

        if (day_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            log.info(
                "Progress: %d/%d days, %d candidates, %.0fs elapsed",
                day_idx + 1, len(sessions), total_candidates, elapsed,
            )

    persistor.flush()
    elapsed = time.time() - t0
    log.info(
        "Complete: %d trading days, %d candidates, %.1fs elapsed",
        len(sessions), total_candidates, elapsed,
    )

    return total_candidates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Replay breakout detection on historical data",
    )
    parser.add_argument("--start-date", type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (default: TradingUniverse)")
    parser.add_argument("--timeframes", type=str, default=None,
                        help="Comma-separated timeframes (default: all)")
    parser.add_argument("--flush-size", type=int, default=200,
                        help="ClickHouse insert batch size (default: 200)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)

    # MDS required for trading sessions (and optionally symbol list)
    md = ServiceLocator.wait_for_service(ServiceLocator.MARKET_DATA, timeout_sec=30)
    log.info("MDS found: %s", md.router)

    sessions = get_trading_sessions(md.router, start, end)
    log.info("Got %d trading sessions from MDS", len(sessions))

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = fetch_symbol_list(md.router, "trading_universe")
        log.info("Using TradingUniverse: %d symbols", len(symbols))

    timeframes = (
        [t.strip() for t in args.timeframes.split(",")]
        if args.timeframes
        else DEFAULT_TIMEFRAMES
    )

    ch = get_ch_client()
    replay(ch, symbols, timeframes, sessions, flush_size=args.flush_size)


if __name__ == "__main__":
    main()
