"""Entry point:  python -m greeks.history <from-date> <to-date> --processes <n> [--dry-run]

Prices historical greeks into option_trade. `--processes` is required (no default): workers are
independent processes, one (day, underlying) each (no asyncio/threading under greeks/). `--dry-run`
prints the pairs and row counts still to do and writes nothing. Built so a nightly step can call it
for a single day; that wiring is a later set.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timezone
from multiprocessing import Pool
from typing import Any, Callable, List, Sequence

from greeks.config import load_config
from greeks.domain import RowStatus
from greeks.history import (
    COLUMNS,
    COLUMNS_NON_GREEKS,
    Deps,
    Pair,
    PairResult,
    Row,
    Summary,
    dataset_for,
    price_rows,
    run,
)

log = logging.getLogger(__name__)
TABLE = "trading.option_trade"
LEDGER = "trading.option_archive_ledger"


# ── ClickHouse IO (production) ───────────────────────────────────────────────────────────────────

def _client() -> Any:
    from greeks.ch import get_ch_client
    return get_ch_client()


def _list_pairs(client: Any, from_date: date, to_date: date) -> List[Pair]:
    res = client.query(
        f"SELECT DISTINCT session_date, underlying FROM {TABLE} "
        "WHERE session_date BETWEEN %(a)s AND %(b)s",
        parameters={"a": from_date, "b": to_date})
    return [(r[0], str(r[1])) for r in res.result_rows]


def _ledgered(client: Any, from_date: date, to_date: date, dataset: str) -> "set[Pair]":
    res = client.query(
        f"SELECT DISTINCT session_date, ticker FROM {LEDGER} FINAL "
        "WHERE dataset = %(d)s AND session_date BETWEEN %(a)s AND %(b)s",
        parameters={"d": dataset, "a": from_date, "b": to_date})
    return {(r[0], str(r[1]).upper()) for r in res.result_rows}


def _count_rows(client: Any, day: date, underlying: str) -> int:
    res = client.query(
        f"SELECT count() FROM {TABLE} FINAL WHERE underlying = %(u)s AND session_date = %(d)s",
        parameters={"u": underlying.upper(), "d": day})
    return int(res.result_rows[0][0]) if res.result_rows else 0


def _read_rows(client: Any, day: date, underlying: str) -> List[Row]:
    cols = ", ".join(COLUMNS_NON_GREEKS)
    res = client.query(
        f"SELECT {cols} FROM {TABLE} FINAL WHERE underlying = %(u)s AND session_date = %(d)s",
        parameters={"u": underlying.upper(), "d": day})
    names = res.column_names
    return [dict(zip(names, row)) for row in res.result_rows]


def _insert_rows(client: Any, rows: Sequence[Row]) -> None:
    if not rows:
        return
    data = [[r[c] for c in COLUMNS] for r in rows]
    client.insert("option_trade", data, column_names=list(COLUMNS), database="trading")


def _write_ledger(client: Any, day: date, underlying: str, n: int, dataset: str) -> None:
    client.insert(
        "option_archive_ledger",
        [[day, underlying.upper(), n, 0, dataset, "", ""]],
        column_names=["session_date", "ticker", "rows_inserted", "quote_failures", "dataset",
                      "gap_reason", "claimed_by"],
        database="trading")


def _process_pair(pair: Pair) -> PairResult:
    """Price one (day, underlying) end to end on this worker's OWN ClickHouse client and config. A
    whole-pair failure (CH/inputs) returns an error (one WARN, no ledger row) so a rerun retries it."""
    day, underlying = pair
    t0 = time.perf_counter()
    try:
        from greeks_server.inputs import clickhouse_loader  # live inputs mechanism (SessionInputs)
        cfg = load_config()
        client = _client()
        loader = clickhouse_loader(lambda: client)
        rows = _read_rows(client, day, underlying)
        inputs = loader(day)
        priced, reasons = price_rows(
            rows, inputs, methodology_version=cfg.methodology_version,
            t_floor_minutes=float(cfg.t_floor_minutes), expiry_time_et=cfg.expiry_time_et,
            now=datetime.now(timezone.utc))
        _insert_rows(client, priced)
        _write_ledger(client, day, underlying, len(priced), dataset_for(cfg.methodology_version))
        ok = sum(1 for r in priced if r["greeks_status"] == RowStatus.SUCCESS.value)
        return PairResult(day, underlying, len(rows), ok, dict(reasons), time.perf_counter() - t0, None)
    except Exception as e:  # noqa: BLE001 — a whole-pair failure; a rerun retries it
        return PairResult(day, underlying, 0, 0, {}, time.perf_counter() - t0, f"{type(e).__name__}: {e}")


def _build_prod_deps(processes: int) -> Deps:
    cfg = load_config()
    dataset = dataset_for(cfg.methodology_version)
    client = _client()  # the driver process's client, for the planning queries only

    def mapper(fn: Callable[[Pair], PairResult], items: List[Pair]) -> List[PairResult]:
        if not items:
            return []
        with Pool(processes=processes) as pool:
            return list(pool.map(fn, items))

    return Deps(
        list_pairs=lambda a, b: _list_pairs(client, a, b),
        ledgered=lambda a, b: _ledgered(client, a, b, dataset),
        count_rows=lambda d, u: _count_rows(client, d, u),
        process_pair=_process_pair,
        mapper=mapper)


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────────

def _parse(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m greeks.history")
    p.add_argument("from_date", type=date.fromisoformat, help="first ET session date (YYYY-MM-DD)")
    p.add_argument("to_date", type=date.fromisoformat, help="last ET session date (YYYY-MM-DD)")
    p.add_argument("--processes", type=int, required=True, help="worker process count (no default)")
    p.add_argument("--dry-run", action="store_true", help="print the pairs/row counts, write nothing")
    return p.parse_args(list(argv))


def main(argv: Sequence[str], *, build_deps: Callable[[int], Deps] = _build_prod_deps,
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
