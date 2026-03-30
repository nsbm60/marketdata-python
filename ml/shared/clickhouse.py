"""
ml/shared/clickhouse.py

ClickHouse connection utilities.
"""

import os

import clickhouse_connect

from discovery import ServiceLocator


def get_ch_client():
    """
    Get ClickHouse client via service discovery.

    Returns a clickhouse_connect client ready to use.
    """
    ch_endpoint = ServiceLocator.wait_for_service(
        ServiceLocator.CLICKHOUSE,
        timeout_sec=60,
    )
    return clickhouse_connect.get_client(
        host=ch_endpoint.host,
        port=ch_endpoint.port,
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", "Aector99"),
        database=os.environ.get("CLICKHOUSE_DATABASE", "trading"),
    )
