"""
The `build` driver: month by month, join the look-ahead-safe inputs (inputs.build_inputs) to the
forward targets (targets.compute), accumulate per calendar year, and write one Parquet per year (temp
file + atomic rename, so a rerun replaces a year cleanly) plus manifest.json. Quiet: one INFO line per
month, nothing per name.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd

from . import DATASET_DIR, DATASET_VERSION, MANIFEST_NAME
from . import inputs as inputs_mod
from . import targets as targets_mod
from .inputs import _COLUMNS as INPUT_COLUMNS

log = logging.getLogger("vol_dataset.dataset")

# The target columns the dataset adds beside the inputs (keys underlying/t/H come from the inputs).
_TARGET_ONLY = ("target_down", "target_up", "realized_down_var", "realized_up_var",
                "implied_total_var", "sample_weight")
FINAL_COLUMNS: Tuple[str, ...] = (*INPUT_COLUMNS, *_TARGET_ONLY)


def _month_ranges(start: dt.date, end: dt.date) -> Iterator[Tuple[dt.date, dt.date]]:
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        nxt = dt.date(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
        yield max(cur, start), min(nxt - dt.timedelta(days=1), end)
        cur = nxt


def _month_frame(ch: Any, ms: dt.date, me: dt.date) -> Tuple[pd.DataFrame, Counter, Counter]:
    """The joined (inputs ⋈ targets) rows for one month, plus that month's drops and coverage."""
    ins = inputs_mod.build_inputs(ch, ms, me)
    tgt, dropped, coverage = targets_mod.compute(ch, ms, me)
    if ins.empty or tgt.empty:
        return pd.DataFrame(columns=list(FINAL_COLUMNS)), dropped, coverage
    merged = ins.merge(tgt, on=["underlying", "t", "H"], how="inner")
    return merged[list(FINAL_COLUMNS)], dropped, coverage


def build(ch: Any, start: dt.date, end: dt.date) -> Dict[str, Any]:
    """Build [start, end] month by month; write each touched year's Parquet and the manifest. Returns
    the manifest dict."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dropped: Counter = Counter()
    coverage: Counter = Counter()
    rows_by_year_h: Counter = Counter()
    by_year: Dict[int, List[pd.DataFrame]] = {}
    years_written: List[int] = []
    dtypes: Dict[str, str] = {}

    def flush(year: int) -> None:
        frames = by_year.pop(year, [])
        year_df = (pd.concat(frames, ignore_index=True) if frames
                   else pd.DataFrame(columns=list(FINAL_COLUMNS)))
        _write_year(year_df, year)
        years_written.append(year)
        if not year_df.empty and not dtypes:
            dtypes.update({c: str(year_df[c].dtype) for c in year_df.columns})

    current_year: int | None = None
    for ms, me in _month_ranges(start, end):
        if current_year is not None and ms.year != current_year:
            flush(current_year)
        current_year = ms.year
        mdf, mdrop, mcov = _month_frame(ch, ms, me)
        dropped.update(mdrop)
        coverage.update(mcov)
        if not mdf.empty:
            by_year.setdefault(ms.year, []).append(mdf)
            for (_y, h), n in mdf.assign(_y=[t.year for t in mdf["t"]]).groupby(
                    ["_y", "H"]).size().items():
                rows_by_year_h[(int(_y), int(h))] += int(n)
        log.info("vol_dataset %s..%s: %d rows", ms, me, len(mdf))
        print(f"[vol_dataset] {ms} .. {me}: {len(mdf):,} rows")
    if current_year is not None:
        flush(current_year)

    manifest = _manifest(start, end, dtypes, rows_by_year_h, dropped, coverage)
    (DATASET_DIR / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[vol_dataset] wrote years {sorted(set(years_written))} to {DATASET_DIR}")
    return manifest


def _write_year(year_df: pd.DataFrame, year: int) -> None:
    """Write one year's Parquet via a temp file + atomic rename (a rerun replaces the year cleanly)."""
    final = DATASET_DIR / f"{year}.parquet"
    tmp = DATASET_DIR / f".{year}.parquet.tmp"
    year_df.to_parquet(tmp, engine="pyarrow", index=False)
    os.replace(tmp, final)


def _manifest(start: dt.date, end: dt.date, dtypes: Dict[str, str], rows_by_year_h: Counter,
              dropped: Counter, coverage: Counter) -> Dict[str, Any]:
    return {
        "version": DATASET_VERSION,
        "git_commit": _git_commit(),
        "build_time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "date_range": {"from": start.isoformat(), "to": end.isoformat()},
        "columns": dtypes or {c: None for c in FINAL_COLUMNS},
        "rows_by_year_h": {f"{y}:{h}": n for (y, h), n in sorted(rows_by_year_h.items())},
        "dropped_rows": dict(sorted(dropped.items())),
        "coverage_incomplete_rv": {f"{y}:{h}": n for (y, h), n in sorted(coverage.items())},
    }


def _git_commit() -> str:
    """The build's commit: `git rev-parse HEAD` in the repo, or the deploy-record next to the app on
    drogon (which carries the committed hash). Raises if neither is available — no silent default."""
    here = Path(__file__).resolve().parent.parent
    try:
        return subprocess.check_output(
            ["git", "-C", str(here), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    record = here.parent / "deploy-record"   # /opt/marketdata-python/deploy-record on drogon
    if record.exists():
        first = record.read_text().splitlines()[0]
        for tok in first.split():
            if len(tok) == 40 and all(c in "0123456789abcdef" for c in tok):
                return tok
    raise RuntimeError(
        f"cannot determine the build commit: `git rev-parse HEAD` failed in {here} and no "
        f"deploy-record with a commit at {record}")
