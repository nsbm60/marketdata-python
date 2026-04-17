"""
ml/models/breakout/persistence.py

Buffered ClickHouse writer for breakout candidates.
"""

import logging
from datetime import date

import clickhouse_connect

from ml.models.breakout.signal_generator import BreakoutCandidate

log = logging.getLogger(__name__)


class BreakoutPersistor:
    """Buffered insert to trading.breakout_candidate."""

    TABLE = "trading.breakout_candidate"
    COLUMNS = [
        "symbol", "timeframe", "ts", "session",
        "direction", "price", "level_price", "level_age_min", "bar_index",
        "ribbon_state", "ribbon_age", "ribbon_spread",
        "atr", "bar_range_atr", "bar_close_pct", "volume_ratio", "gap_pct",
        "score", "source",
        "prior_session_high", "prior_session_low",
    ]

    def __init__(self, ch_client: clickhouse_connect.driver.Client,
                 flush_size: int = 50):
        self._ch = ch_client
        self._flush_size = flush_size
        self._buffer: list[list] = []

    def persist(self, candidate: BreakoutCandidate, timeframe: str,
                session_date: date, source: str = "live") -> None:
        """Buffer a candidate for insert. Auto-flushes at flush_size."""
        row = [
            candidate.symbol,
            timeframe,
            candidate.ts,
            session_date,
            candidate.direction.value,       # "long" or "short"
            candidate.price,
            candidate.level_price,
            candidate.level_age_minutes,
            candidate.bar_index,
            candidate.ribbon_state.value,    # "ordered_bullish" etc.
            candidate.ribbon_age,
            candidate.ribbon_spread_pct,
            candidate.atr,
            candidate.bar_range_atr,
            candidate.bar_close_pct,
            candidate.volume_ratio,
            candidate.gap_pct,
            candidate.score,
            source,
            candidate.prior_session_high,
            candidate.prior_session_low,
        ]
        self._buffer.append(row)

        if len(self._buffer) >= self._flush_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered rows to ClickHouse."""
        if not self._buffer:
            return
        count = len(self._buffer)
        try:
            self._ch.insert(self.TABLE, self._buffer, column_names=self.COLUMNS)
            log.info("Flushed %d candidates to %s", count, self.TABLE)
        except Exception as e:
            log.error("Failed to flush %d candidates: %s", count, e)
        finally:
            # Rows intentionally dropped on error — this is a research tool,
            # not a trading system. Re-run the replay to regenerate.
            self._buffer.clear()
