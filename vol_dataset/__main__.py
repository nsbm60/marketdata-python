"""
The vol-dataset commands (roadmap-2026-11-16 Stage 4).

    python -m vol_dataset build <from> <to>   # write the years touched; one line per month
    python -m vol_dataset summary             # read the files only; print the dataset proof

build reads ClickHouse and writes Parquet under DATASET_DIR; it creates no ClickHouse tables.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.shared.clickhouse import get_ch_client  # noqa: E402
from vol_dataset import dataset, summary as summary_mod  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(prog="vol_dataset")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("start", metavar="from", help="first t, YYYY-MM-DD (inclusive)")
    b.add_argument("end", metavar="to", help="last t, YYYY-MM-DD (inclusive)")
    sub.add_parser("summary")
    args = p.parse_args()

    if args.cmd == "build":
        start = dt.date.fromisoformat(args.start)
        end = dt.date.fromisoformat(args.end)
        ch: Any = get_ch_client()
        print(f"[vol_dataset] build {start} .. {end}")
        dataset.build(ch, start, end)
    elif args.cmd == "summary":
        summary_mod.summary()


if __name__ == "__main__":
    main()
