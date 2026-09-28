"""CLI: seed the work queue from the watchlist, oldest-first. One-time (idempotent).

    python -m option_archive.seed

Enumerates every watchlist underlying over [backfill_start_date, today] at its band
(SPY/QQQ use the ±10% override), and enqueues (contract, day) jobs. INSERT OR IGNORE,
so re-running only adds new jobs. Reads the watchlist from ClickHouse and contract
reference / spot from Massive + Alpaca; writes only to the SQLite queue.
"""

from __future__ import annotations

import logging
from datetime import date

from greeks.pull.alpaca_spot import make_stock_client
from ml.shared.clickhouse import get_ch_client  # canonical connector (carries the CH password)
from option_archive.config import get_config
from option_archive.queue import WorkQueue
from option_archive.reference import seed_watchlist

log = logging.getLogger("option_archive.seed")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = get_config()
    ch = get_ch_client()
    alpaca = make_stock_client(cfg.api_keys.alpaca_api_key, cfg.api_keys.alpaca_api_secret)
    with WorkQueue(
        cfg.queue_db_path,
        lease=cfg.queue.lease,
        max_attempts=cfg.queue.max_attempts,
        backoff_base=cfg.queue.backoff_base,
    ) as queue:
        report = seed_watchlist(
            queue, ch, cfg,
            massive_api_key=cfg.api_keys.massive_api_key,
            alpaca=alpaca,
            now=date.today(),
        )
    log.info(
        "SEED COMPLETE: underlyings=%d trading_days=%d eligible_contract_days=%d "
        "enqueued=%d nonstandard_excluded=%d days_without_bar=%d",
        report.underlyings, report.trading_days, report.eligible_contract_days,
        report.enqueued, report.nonstandard_excluded, report.days_without_bar,
    )


if __name__ == "__main__":
    main()
