"""Column contract against the real Scala DDL (system of record, per jumps/SCHEMA.md) — the same
pattern tests/iv_event/test_event_columns.py uses, for trading.jump_daily."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pytest

from jumps.jump_daily import COLUMNS


def _canonical_ddl_path() -> Path:
    schema_md = (Path(__file__).resolve().parents[2] / "jumps" / "SCHEMA.md").read_text(encoding="utf-8")
    m = re.search(r"Canonical DDL\s*\|\s*`([^`]+database_schema\.sql)`", schema_md)
    assert m is not None, "jumps/SCHEMA.md has no Canonical DDL path"
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
        pytest.fail(f"jump_daily DDL not found at {path} (system of record per jumps/SCHEMA.md)")
    return path.read_text(encoding="utf-8")


def test_writer_columns_match_the_jump_daily_ddl() -> None:
    ddl_cols = _table_columns(_ddl(), "trading.jump_daily")
    # The writer provides every DDL column, computed_at included (no DEFAULT on it in the DDL).
    assert set(ddl_cols) == set(COLUMNS)
    assert "computed_at" in COLUMNS
    # Insert order matches DDL order (the INSERT … SELECT relies on this).
    assert list(COLUMNS) == ddl_cols
