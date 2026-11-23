"""Column contracts against the real Scala DDL (system of record, per iv_event/SCHEMA.md) — the same
pattern tests/iv_curve uses for iv_curve_daily, for both iv_event tables."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pytest

from iv_event.daytype import _DAYTYPE_COLUMNS
from iv_event.event import COLUMNS
from iv_event.outcome_event import OUTCOME_COLUMNS


def _canonical_ddl_path() -> Path:
    schema_md = (Path(__file__).resolve().parents[2] / "iv_event" / "SCHEMA.md").read_text(encoding="utf-8")
    m = re.search(r"Canonical DDL\s*\|\s*`([^`]+database_schema\.sql)`", schema_md)
    assert m is not None, "iv_event/SCHEMA.md has no Canonical DDL path"
    return Path(m.group(1)).expanduser()


def _table_columns(ddl: str, table: str) -> List[str]:
    start = ddl.find(f"CREATE TABLE IF NOT EXISTS {table}")
    assert start != -1, f"{table} CREATE not found in the DDL"
    body = ddl[ddl.index("(", start) + 1:]
    cols: List[str] = []
    for line in body.splitlines():
        if line.lstrip().startswith(")"):
            break
        m = re.match(r"\s+(\w+)\s+\w", line)  # a column definition: name followed by a type token
        if m:
            cols.append(m.group(1))
    return cols


def _ddl() -> str:
    path = _canonical_ddl_path()
    if not path.exists():
        pytest.fail(f"iv_event DDL not found at {path} (system of record per iv_event/SCHEMA.md)")
    return path.read_text(encoding="utf-8")


def test_window_writer_columns_match_the_iv_event_window_ddl() -> None:
    ddl_cols = _table_columns(_ddl(), "trading.iv_event_window")
    # The writer writes every DDL column except computed_at (the ReplacingMergeTree version DEFAULT).
    assert set(ddl_cols) - {"computed_at"} == set(COLUMNS)
    assert "computed_at" not in COLUMNS
    assert "computed_at" in ddl_cols


def test_daytype_writer_columns_match_the_iv_event_daytype_ddl() -> None:
    ddl_cols = _table_columns(_ddl(), "trading.iv_event_daytype")
    assert set(ddl_cols) - {"computed_at"} == set(_DAYTYPE_COLUMNS)
    assert "computed_at" not in _DAYTYPE_COLUMNS
    assert "computed_at" in ddl_cols


def test_outcome_writer_columns_match_the_iv_event_outcome_ddl() -> None:
    ddl_cols = _table_columns(_ddl(), "trading.iv_event_outcome")
    # The writer provides every DDL column, computed_at included (no DEFAULT on it in the DDL), and
    # the insert order matches DDL order.
    assert set(ddl_cols) == set(OUTCOME_COLUMNS)
    assert "computed_at" in OUTCOME_COLUMNS
    assert list(OUTCOME_COLUMNS) == ddl_cols
