"""
ml/shared/clickhouse.py

ClickHouse connection utilities.
"""

import logging
import os
import time

import clickhouse_connect

from discovery import ServiceLocator

log = logging.getLogger("ml.shared.clickhouse")

# Discovery is the single source of connection truth for every process, batch
# included (no env override). If the locator isn't answering yet, poll it with
# capped backoff and a clear journal line rather than crash — the same idle-poll
# posture we take toward the network. Boot ordering is the unit's After=/Wants=.
_RESOLVE_ATTEMPT_TIMEOUT_SEC = 30.0
_RESOLVE_BACKOFF_MAX_SEC = 30.0


def get_ch_client():
    """
    Get ClickHouse client via service discovery.

    Discovery is the single source of connection truth for every process, batch
    included — there is no env override. If the ServiceLocator is not answering
    yet (e.g. a batch box that booted ahead of the ClickHouse discovery sidecar),
    idle-poll with capped backoff, logging "waiting for ServiceLocator", instead
    of crashing — the same posture we take toward the network. Boot ordering is
    the unit's After=/Wants=, not this code's job.

    Returns a clickhouse_connect client ready to use.
    """
    backoff = 1.0
    while True:
        try:
            ch_endpoint = ServiceLocator.wait_for_service(
                ServiceLocator.CLICKHOUSE,
                timeout_sec=_RESOLVE_ATTEMPT_TIMEOUT_SEC,
            )
            break
        except TimeoutError:
            log.warning(
                "waiting for ServiceLocator to advertise clickhouse; "
                "retrying in %.0fs",
                backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, _RESOLVE_BACKOFF_MAX_SEC)
    return clickhouse_connect.get_client(
        host=ch_endpoint.host,
        port=ch_endpoint.port,
        username=os.environ.get("CLICKHOUSE_USER") or "default",
        password=os.environ.get("CLICKHOUSE_PASSWORD") or "Aector99",
        database=os.environ.get("CLICKHOUSE_DATABASE") or "trading",
    )
