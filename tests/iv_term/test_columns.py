"""Column contract against the real Scala DDL (system of record, per iv_term/SCHEMA.md) — the same
pattern tests/greeks uses for option_trade."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pytest

from iv_term.term import COLUMNS


def _canonical_ddl_path() -> Path:
    """The DDL path recorded in iv_term/SCHEMA.md — the system of record is the Scala repo, not a
    copy here."""
    schema_md = (Path(__file__).resolve().parents[2] / "iv_term" / "SCHEMA.md").read_text(encoding="utf-8")
    m = re.search(r"Canonical DDL\s*\|\s*`([^`]+database_schema\.sql)`", schema_md)
    assert m is not None, "iv_term/SCHEMA.md has no Canonical DDL path"
    return Path(m.group(1)).expanduser()


def _iv_term_columns(ddl: str) -> List[str]:
    """The column names in the trading.iv_term_daily CREATE block."""
    start = ddl.find("CREATE TABLE IF NOT EXISTS trading.iv_term_daily")
    assert start != -1, "trading.iv_term_daily CREATE not found in the DDL"
    body = ddl[ddl.index("(", start) + 1:]
    cols: List[str] = []
    for line in body.splitlines():
        if line.lstrip().startswith(")"):
            break
        m = re.match(r"\s+(\w+)\s+\w", line)  # a column definition: name followed by a type token
        if m:
            cols.append(m.group(1))
    return cols


def test_writer_columns_match_the_iv_term_daily_ddl() -> None:
    path = _canonical_ddl_path()
    if not path.exists():
        pytest.fail(f"iv_term_daily DDL not found at {path} (system of record per iv_term/SCHEMA.md)")
    ddl_cols = _iv_term_columns(path.read_text(encoding="utf-8"))
    # The writer writes every DDL column except computed_at (the ReplacingMergeTree version DEFAULT).
    assert set(ddl_cols) - {"computed_at"} == set(COLUMNS)
    assert "computed_at" not in COLUMNS
    # The DDL really does carry the version column (guards against it being dropped).
    assert "computed_at" in ddl_cols
