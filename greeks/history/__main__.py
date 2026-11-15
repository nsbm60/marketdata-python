"""Entry point:  python -m greeks.history <from-date> <to-date> --processes <n> [--dry-run]

Prices historical greeks into option_trade. `--processes` is required (no default): workers are
independent processes, one (day, underlying) each (no asyncio/threading under greeks/). `--dry-run`
prints the pairs and row counts still to do and writes nothing. Built so a nightly step can call it
for a single day; that wiring is a later set.

This module only parses arguments and calls into greeks.history.worker — the worker function must
live in an importable module (not __main__) so forkserver workers can import it by module path.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from typing import Callable, Sequence

from greeks.history import Deps, Summary, run
from greeks.history.worker import default_build_deps


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m greeks.history")
    p.add_argument("from_date", type=date.fromisoformat, help="first ET session date (YYYY-MM-DD)")
    p.add_argument("to_date", type=date.fromisoformat, help="last ET session date (YYYY-MM-DD)")
    p.add_argument("--processes", type=int, required=True, help="worker process count (no default)")
    p.add_argument("--dry-run", action="store_true", help="print the pairs/row counts, write nothing")
    return p.parse_args(list(argv))


def main(argv: Sequence[str], *, build_deps: Callable[[int], Deps] = default_build_deps,
         run_fn: Callable[[date, date, bool, Deps], Summary] = run) -> int:
    args = _parse(argv)
    if args.processes < 1:
        raise SystemExit("--processes must be >= 1")
    deps = build_deps(args.processes)
    run_fn(args.from_date, args.to_date, args.dry_run, deps)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s [%(processName)s] %(name)s - %(message)s")
    raise SystemExit(main(sys.argv[1:]))
