"""The importable worker + ClickHouse IO + process pool for greeks.history.

This lives in a real module (NOT __main__): Python 3.14 makes forkserver the default start method on
Linux, and a forkserver worker imports its task BY MODULE PATH — a function defined in __main__ can't
be imported that way (every worker died with AttributeError: module '__main__' has no attribute
'_process_pair'). So the worker function and everything it needs live here; __main__ only parses args.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import date, datetime, timezone
from typing import Any, Callable, Iterable, Iterator, List, Sequence

from greeks.config import load_config
from greeks.domain import RowStatus
from greeks.history import COLUMNS, COLUMNS_NON_GREEKS, Deps, Pair, PairResult, Row, dataset_for, price_rows

TABLE = "trading.option_trade"
LEDGER = "trading.option_archive_ledger"

# Mapper type: run a per-pair function over the pairs and STREAM the results as they complete.
Mapper = Callable[[Callable[[Pair], PairResult], List[Pair]], Iterable[PairResult]]

# Set to "ok" / "die" to run the entry point against in-process fakes over the real forkserver pool
# (the subprocess entry-point test), instead of touching ClickHouse.
SELFTEST_ENV = "GREEKS_HISTORY_SELFTEST"


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
    """Price one (day, underlying) end to end on this worker's OWN ClickHouse client and config.
    Importable by module path (greeks.history.worker._process_pair) so forkserver can run it. A
    whole-pair failure (CH/inputs) returns an error result (one WARN, no ledger row); a DEAD worker is
    a different thing — that stops the run via BrokenProcessPool in the mapper."""
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


# ── process pool (forkserver, fail-fast) ─────────────────────────────────────────────────────────

def pool_mapper(processes: int) -> Mapper:
    """Run each pair in its own forkserver worker. A dead worker (anything other than a per-pair
    failure, which returns an error result) stops the run at once with a specific error — never the
    silent respawn-dead-workers loop a multiprocessing.Pool does."""
    # forkserver, stated explicitly: Python 3.14 made it the Linux default, and under it a worker
    # imports its task by module path — so the start method can't drift and break the import again.
    ctx = multiprocessing.get_context("forkserver")

    def mapper(fn: Callable[[Pair], PairResult], items: List[Pair]) -> Iterator[PairResult]:
        if not items:
            return
        # One executor for the whole run; yield each pair's result the moment it completes (as_completed,
        # so possibly out of order) rather than waiting for all of them. A dead worker still raises
        # BrokenProcessPool here, stopping the run with the specific error.
        with ProcessPoolExecutor(max_workers=processes, mp_context=ctx) as ex:
            futures = [ex.submit(fn, it) for it in items]
            for f in as_completed(futures):
                try:
                    yield f.result()
                except BrokenProcessPool as e:
                    raise RuntimeError(
                        f"greeks.history worker process died — stopping the run: {e}") from e

    return mapper


def _build_prod_deps(processes: int) -> Deps:
    cfg = load_config()
    dataset = dataset_for(cfg.methodology_version)
    client = _client()  # the driver process's client, for the planning queries only
    return Deps(
        list_pairs=lambda a, b: _list_pairs(client, a, b),
        ledgered=lambda a, b: _ledgered(client, a, b, dataset),
        count_rows=lambda d, u: _count_rows(client, d, u),
        process_pair=_process_pair,
        mapper=pool_mapper(processes))


# ── self-test hooks (the subprocess entry-point test; no ClickHouse) ─────────────────────────────

def _selftest_pair(pair: Pair) -> PairResult:
    """A worker that does no ClickHouse work — proves forkserver imports and runs the task."""
    return PairResult(pair[0], pair[1], rows=1, priced=1, reasons={}, seconds=0.0, error=None)


def _suicide_pair(pair: Pair) -> PairResult:
    os._exit(1)  # die without returning — a worker death, not a per-pair failure


def _selftest_deps(processes: int, pair_fn: Callable[[Pair], PairResult]) -> Deps:
    pairs = [(date(2016, 1, 4), "AAA"), (date(2016, 1, 5), "BBB")]
    return Deps(
        list_pairs=lambda a, b: list(pairs),
        ledgered=lambda a, b: set(),
        count_rows=lambda d, u: 0,
        process_pair=pair_fn,
        mapper=pool_mapper(processes))


def default_build_deps(processes: int) -> Deps:
    """Production deps, unless SELFTEST_ENV selects an in-process self-test over the real pool."""
    mode = os.environ.get(SELFTEST_ENV)
    if mode == "ok":
        return _selftest_deps(processes, _selftest_pair)
    if mode == "die":
        return _selftest_deps(processes, _suicide_pair)
    return _build_prod_deps(processes)
