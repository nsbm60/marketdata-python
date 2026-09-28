"""CLI: drain the trade queue via the per-day flat-file ingest.

    python -m option_archive            # run one worker until the queue drains

Credentials: the flat-file S3 key/secret come from env
(``MASSIVE_S3_ACCESS_KEY`` / ``MASSIVE_S3_SECRET_KEY``); Alpaca keys from
config/env; ClickHouse via the canonical ``ml.shared.clickhouse.get_ch_client``
connector, resolved through service discovery (the locator), same as every other
process. Queue policy (lease, retries, backoff) and the queue DB path come from
config, not flags.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta

from greeks.pull.alpaca_spot import make_stock_client
from ml.shared.clickhouse import get_ch_client  # canonical connector (discovery for all)
from option_archive.config import get_config
from option_archive.ingest_day import make_s3_client, run_worker
from option_archive.queue import WorkQueue

log = logging.getLogger("option_archive")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # httpx logs each request URL at INFO, and vendor URLs carry the API key as a
    # query param — keep httpx at WARNING so the key never reaches the journal.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = get_config()
    s3 = make_s3_client(
        os.environ.get("MASSIVE_S3_ACCESS_KEY", ""),
        os.environ.get("MASSIVE_S3_SECRET_KEY", ""),
        connect_timeout=cfg.s3.connect_timeout,
        read_timeout=cfg.s3.read_timeout,
    )
    alpaca = make_stock_client(cfg.api_keys.alpaca_api_key, cfg.api_keys.alpaca_api_secret)
    ch = get_ch_client()
    with WorkQueue(
        cfg.queue_db_path,
        lease=cfg.queue.lease,
        max_attempts=cfg.queue.max_attempts,
        backoff_base=cfg.queue.backoff_base,
    ) as queue:
        report = run_worker(queue, ch, s3, alpaca, cfg, poll=timedelta(seconds=30))
    log.info("ingest complete: days=%d trades=%d", report.days_done, report.trades_inserted)


if __name__ == "__main__":
    main()
