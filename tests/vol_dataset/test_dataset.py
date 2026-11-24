"""The build orchestration (no ClickHouse): whole-year expansion, and the per-year manifest merge —
a rebuild of one year must leave another year's manifest entry (and Parquet) intact."""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter

import pandas as pd

from vol_dataset import dataset


def _fake_month_frame(_ch, ms: dt.date, _me: dt.date):
    """A tiny deterministic month: two rows (H 5 and 10), one dropped corp action, two coverage H5."""
    df = pd.DataFrame(
        {"underlying": ["AAA", "AAA"], "t": [ms, ms], "H": [5, 10], "target_down": [0.1, 0.2]})
    return df, Counter({"corporate_action": 1}), Counter({(ms.year, 5): 2})


def _setup(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(dataset, "_month_frame", _fake_month_frame)
    monkeypatch.setattr(dataset, "_git_commit", lambda: "deadbeef" * 5)   # 40 hex chars


def _manifest(tmp_path) -> dict:
    return json.loads((tmp_path / "manifest.json").read_text())


def test_build_expands_to_whole_year(monkeypatch, tmp_path) -> None:
    _setup(monkeypatch, tmp_path)
    # a mid-year range must build the whole calendar year.
    dataset.build(object(), dt.date(2023, 3, 15), dt.date(2023, 7, 20))
    assert (tmp_path / "2023.parquet").exists()
    man = _manifest(tmp_path)
    assert set(man["years"]) == {"2023"}
    # 12 months each contributed one H5 and one H10 row.
    assert man["years"]["2023"]["rows_by_h"] == {"5": 12, "10": 12}
    assert man["years"]["2023"]["dropped_rows"] == {"corporate_action": 12}
    assert man["years"]["2023"]["coverage_incomplete_rv"] == {"5": 24}


def test_rebuild_one_year_keeps_another_years_entry_and_file(monkeypatch, tmp_path) -> None:
    _setup(monkeypatch, tmp_path)
    dataset.build(object(), dt.date(2023, 1, 1), dt.date(2023, 12, 31))
    man1 = _manifest(tmp_path)
    entry_2023 = man1["years"]["2023"]
    assert (tmp_path / "2023.parquet").exists()

    # Rebuild only 2024. 2023's entry and file must be untouched; 2024 added.
    dataset.build(object(), dt.date(2024, 5, 1), dt.date(2024, 6, 1))
    man2 = _manifest(tmp_path)
    assert set(man2["years"]) == {"2023", "2024"}
    assert man2["years"]["2023"] == entry_2023          # byte-for-byte the first build's entry
    assert (tmp_path / "2023.parquet").exists()
    assert (tmp_path / "2024.parquet").exists()
    # the manifest is written atomically — no temp file left behind.
    assert not (tmp_path / "manifest.json.tmp").exists()


def test_rebuild_same_year_replaces_its_entry(monkeypatch, tmp_path) -> None:
    _setup(monkeypatch, tmp_path)
    dataset.build(object(), dt.date(2023, 1, 1), dt.date(2023, 3, 31))   # -> whole 2023
    first = _manifest(tmp_path)["years"]["2023"]["rows_by_h"]
    dataset.build(object(), dt.date(2023, 6, 1), dt.date(2023, 6, 30))   # -> whole 2023 again
    second = _manifest(tmp_path)["years"]["2023"]["rows_by_h"]
    assert first == second == {"5": 12, "10": 12}       # replaced, not doubled
