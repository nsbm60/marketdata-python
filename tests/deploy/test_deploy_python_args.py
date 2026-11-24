"""Argument handling of ./deploy-python, exercised in dry-run mode (no git, no ssh).

DEPLOY_PYTHON_DRY_RUN=1 makes deploy-python validate the host and the --restart flags and then exit
0 before any git archive / ssh. Bad arguments still exit non-zero at validation. These cases pin:
a missing host, an unknown host, a --restart on balerion (which has no services), and a --restart of
an unknown service on drogon."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy-python"


def _run(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "DEPLOY_PYTHON_DRY_RUN": "1"}
    return subprocess.run(["bash", str(DEPLOY), *args], capture_output=True, text=True, env=env)


def test_missing_host_rejected_with_accepted_list() -> None:
    r = _run()
    assert r.returncode != 0
    assert "accepted hosts:" in r.stderr
    assert "drogon" in r.stderr and "balerion" in r.stderr


def test_unknown_host_rejected_with_accepted_list() -> None:
    r = _run("mars")
    assert r.returncode != 0
    assert "unknown host: mars" in r.stderr
    assert "drogon" in r.stderr and "balerion" in r.stderr


def test_balerion_restart_rejected_no_services() -> None:
    r = _run("balerion", "--restart", "pubsub-broker")
    assert r.returncode != 0
    assert "not a managed service on balerion" in r.stderr


def test_drogon_restart_unknown_service_rejected() -> None:
    r = _run("drogon", "--restart", "no-such-svc")
    assert r.returncode != 0
    assert "not a managed service on drogon" in r.stderr


def test_drogon_good_restart_dry_run_ok() -> None:
    r = _run("drogon", "--restart", "pubsub-broker")
    assert r.returncode == 0, r.stderr
    assert "[dry-run] host=drogon" in r.stdout
    assert "restart=[--restart pubsub-broker]" in r.stdout


def test_balerion_no_services_dry_run_shows_ml_extra() -> None:
    r = _run("balerion")
    assert r.returncode == 0, r.stderr
    assert "host=balerion" in r.stdout
    assert "services=[]" in r.stdout
    assert "extras=[ml]" in r.stdout


def test_drogon_restart_all_dry_run_ok() -> None:
    r = _run("drogon", "--restart-all")
    assert r.returncode == 0, r.stderr
    assert "host=drogon" in r.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
