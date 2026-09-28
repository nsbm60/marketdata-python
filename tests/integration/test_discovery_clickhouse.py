#!/usr/bin/env python3
"""Integration regression pin for the ClickHouse discovery→connect path.

Resolves ClickHouse via the REAL discovery-router (no mocks), prints the resolved
host/port/user and a REDACTED password (length + first/last char only — never the
value), then connects through the canonical connector and runs SELECT 1.

This exists because the option_archive seed hit ClickHouse error 194 (auth failed,
user `default`) while live Scala services authenticate fine through the same
locator. The ClickHouse advertisement carries only host+port (see
ClickHouseBroadcaster.scala) — credentials are client-side — so the mismatch is in
what THIS path sends. Compare the printed resolution against what the live services
use; the mismatch it prints is the bug.

Run standalone (e.g. on drogon, as the mdapps service user with its env loaded):
    python tests/integration/test_discovery_clickhouse.py
or under pytest (skips cleanly when no locator/broadcaster is reachable):
    pytest tests/integration/test_discovery_clickhouse.py -s

Requires discovery (the discovery-router) reachable and ClickHouse advertised.
"""
import os
import sys

import pytest

try:
    from discovery.service_locator import ServiceEndpoint, ServiceLocator
    from ml.shared.clickhouse import get_ch_client
except ModuleNotFoundError:
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from discovery.service_locator import ServiceEndpoint, ServiceLocator
    from ml.shared.clickhouse import get_ch_client

# Short: on the discovery bus ClickHouse is advertised ~1/sec, so resolution is
# immediate when the locator is up; a short timeout means a quick skip when it isn't.
_LOCATOR_TIMEOUT_SEC = 10.0


def _redact(value: "str | None") -> str:
    """Password disclosure limited to length + first/last char, per the debugging
    ask. The set-but-empty case is called out because ``os.environ.get(k, default)``
    returns "" (not the default) when the var is present but blank — an empty env
    var silently overrides the connector's built-in default."""
    if value is None:
        return "<unset — connector uses its built-in default>"
    if value == "":
        return "<SET BUT EMPTY, len=0>  <-- empty env var OVERRIDES the default → sends blank password"
    if len(value) <= 2:
        return f"len={len(value)} (too short to show ends safely)"
    return f"len={len(value)} '{value[0]}…{value[-1]}'"


def _env_source(key: str, default: str) -> str:
    return f"{os.environ.get(key, default)} ({'from env' if key in os.environ else 'connector default'})"


def _report(ep: ServiceEndpoint) -> None:
    print("\n=== ClickHouse resolution (via the real discovery-router) ===")
    print(f"  DISCOVERY_HOST/PORT : {os.environ.get('DISCOVERY_HOST', 'localhost')}:"
          f"{os.environ.get('DISCOVERY_PORT', '6005')}")
    print(f"  discovered host     : {ep.host}")
    print(f"  discovered port     : {ep.port}")
    print(f"  CLICKHOUSE_USER     : {_env_source('CLICKHOUSE_USER', 'default')}")
    print(f"  CLICKHOUSE_DATABASE : {_env_source('CLICKHOUSE_DATABASE', 'trading')}")
    print(f"  CLICKHOUSE_PASSWORD : {_redact(os.environ.get('CLICKHOUSE_PASSWORD'))}")
    print("=============================================================\n")


def _resolve(timeout_sec: float) -> ServiceEndpoint:
    return ServiceLocator.wait_for_service(ServiceLocator.CLICKHOUSE, timeout_sec=timeout_sec)


def test_discovery_clickhouse_select_1() -> None:
    """Resolve via discovery, report the redacted resolution, connect, SELECT 1."""
    try:
        ep = _resolve(_LOCATOR_TIMEOUT_SEC)
    except TimeoutError:
        pytest.skip(
            f"no discovery-router reachable (ClickHouse not advertised within "
            f"{_LOCATOR_TIMEOUT_SEC:.0f}s) — run on a host on the discovery bus"
        )
    _report(ep)
    client = get_ch_client()  # the real canonical path, incl. its own resolution + retry
    rows = client.query("SELECT 1").result_rows
    assert rows == [(1,)], f"SELECT 1 returned {rows!r}"
    print("SELECT 1 OK — canonical connector authenticated against ClickHouse.")


if __name__ == "__main__":
    try:
        ep = _resolve(_LOCATOR_TIMEOUT_SEC)
    except TimeoutError:
        print(f"SKIP: no discovery-router reachable within {_LOCATOR_TIMEOUT_SEC:.0f}s "
              "— run where the discovery bus is reachable (e.g. drogon).")
        sys.exit(2)
    _report(ep)
    try:
        client = get_ch_client()
        rows = client.query("SELECT 1").result_rows
    except Exception as e:  # noqa: BLE001 — standalone diagnostic, surface the raw error
        print(f"CONNECT/AUTH FAILED: {type(e).__name__}: {e}")
        print("(compare the redacted resolution above against what the live services use)")
        sys.exit(1)
    ok = rows == [(1,)]
    print(f"SELECT 1 -> {rows}  ({'OK' if ok else 'UNEXPECTED'})")
    sys.exit(0 if ok else 1)
