"""
The read-only `summary`: reads the written Parquet files and manifest.json only (never ClickHouse).
Prints rows per year and H, the NULL share per column, target_down/target_up quantiles per H, the
sample-weight quantiles, the dropped rows by reason, and the coverage (ok-curve rows with incomplete
H rv days) per year and H.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from . import DATASET_DIR, MANIFEST_NAME

_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def summary(root: Path = DATASET_DIR) -> None:
    files = sorted(root.glob("[0-9][0-9][0-9][0-9].parquet"))
    print(f"vol_dataset summary — {root}")
    if not files:
        print("  (no Parquet files)")
        return
    frames: List[pd.DataFrame] = [pd.read_parquet(f, engine="pyarrow") for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["_year"] = [t.year for t in df["t"]]

    print(f"\n  rows per year and H ({len(df):,} total):")
    print(f"    {'year':>6s} " + " ".join(f"{('H' + str(h)):>10s}" for h in _horizons(df)))
    for y, grp in df.groupby("_year"):
        counts = grp.groupby("H").size()
        print(f"    {int(y):>6d} " + " ".join(f"{int(counts.get(h, 0)):>10,d}" for h in _horizons(df)))

    print("\n  NULL share per column:")
    for col in df.columns:
        if col == "_year":
            continue
        print(f"    {col:<42s} {df[col].isna().mean() * 100:>6.2f}%")

    print("\n  target_down / target_up quantiles (5/25/50/75/95) per H:")
    for h in _horizons(df):
        g = df[df["H"] == h]
        qd = g["target_down"].quantile(_QUANTILES)
        qu = g["target_up"].quantile(_QUANTILES)
        print(f"    H{h:<3d} down " + " ".join(f"{qd.loc[q]:>8.3f}" for q in _QUANTILES))
        print(f"    H{h:<3d} up   " + " ".join(f"{qu.loc[q]:>8.3f}" for q in _QUANTILES))

    wq = df["sample_weight"].quantile(_QUANTILES)
    print("\n  sample_weight quantiles (5/25/50/75/95): "
          + " ".join(f"{wq.loc[q]:>10.1f}" for q in _QUANTILES))

    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text())
        years = man.get("years", {})
        print("\n  dropped rows per year (manifest):")
        for year in sorted(years):
            dr = years[year].get("dropped_rows", {})
            parts = " ".join(f"{reason}={int(n):,}" for reason, n in sorted(dr.items())) or "none"
            print(f"    {year}: {parts}  (built {years[year].get('git_commit', '?')[:12]} "
                  f"{years[year].get('build_time', '?')})")
        print("  coverage per year — ok curve, incomplete H rv days (H -> count):")
        for year in sorted(years):
            cov = years[year].get("coverage_incomplete_rv", {})
            parts = " ".join(f"H{h}={int(n):,}" for h, n in sorted(cov.items(), key=lambda kv: int(kv[0]))) or "none"
            print(f"    {year}: {parts}")
    else:
        print("\n  (no manifest.json — dropped rows and coverage unavailable)")


def _horizons(df: pd.DataFrame) -> List[int]:
    return sorted(int(h) for h in df["H"].unique())
