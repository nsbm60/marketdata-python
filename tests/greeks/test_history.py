"""greeks.history: the trade-basis pricer shares the live core, stores each failure reason with null
greeks, copies every non-greeks column exactly, matches the option_trade DDL column contract, skips
ledgered pairs, writes nothing on a dry run, and parses its CLI. No live ClickHouse, no processes.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import pytest

from greeks.domain import Dividend
from greeks.history import (
    COLUMNS,
    COLUMNS_NON_GREEKS,
    Deps,
    PairResult,
    dataset_for,
    price_rows,
    run,
)
from greeks.history.__main__ import main
from greeks_server.compute import Contract, Methodology, compute
from greeks_server.inputs import SessionInputs
from greeks_server.price import PriceKind, SelectedPrice

SESSION = date(2026, 10, 20)
NOW = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)  # 11:00 ET, inside the session
NOW_MS = int(NOW.timestamp() * 1000)
M = Methodology(methodology_version="v2", t_floor_minutes=15.0, expiry_time_et="16:00")
MV, TFLOOR, ETIME = "v2", 15.0, "16:00"


def _sofr() -> Dict[date, float]:
    return {SESSION - timedelta(days=i): 0.043 for i in range(1, 30)}


def _inputs(dividends: Dict[str, tuple[Dividend, ...]] | None = None) -> SessionInputs:
    return SessionInputs(session_date=SESSION, sofr=_sofr(), dividends=dividends or {})


def _row(symbol: str = "NVDA261120C00180000", price: float = 9.1,
         spot: float | None = 181.0, **over: Any) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        "symbol": symbol, "underlying": "NVDA", "session_date": SESSION, "trade_ts": NOW,
        "price": Decimal(str(price)), "size": 3.0, "exchange": 5, "conditions": [227, 14],
        "sip_timestamp_ns": 1_700_000_000_000_000_000, "ordinal": 0, "sequence_number": 99,
        "participant_timestamp_ns": 1_699_999_999_999_999_990, "correction": 0,
        "spot_at_trade": None if spot is None else Decimal(str(spot)), "spot_trade_ts": NOW,
        "quote_ts": None, "bid": None, "ask": None, "bid_size": None, "ask_size": None,
        "quote_lag_ms": None, "source": "massive",
    }
    r.update(over)
    return r


def _price(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    priced, _ = price_rows(rows, _inputs(), methodology_version=MV, t_floor_minutes=TFLOOR,
                           expiry_time_et=ETIME, now=NOW)
    return priced


# ── one shared core ──────────────────────────────────────────────────────────────────────────────

def test_live_quote_mid_and_historical_trade_go_through_the_same_core() -> None:
    inp = _inputs()
    c = Contract.parse("NVDA261120C00180000")
    live = compute(c, SelectedPrice(9.1, PriceKind.MID, NOW_MS),
                   SelectedPrice(181.0, PriceKind.MID, NOW_MS), inp, NOW, M)
    assert live.status.value == "success" and live.iv is not None
    h = _price([_row(price=9.1, spot=181.0)])[0]
    assert h["greeks_status"] == "success" and h["greeks_basis"] == "trade"
    assert h["iv"] == pytest.approx(live.iv)
    assert h["delta"] == pytest.approx(live.delta) and h["gamma"] == pytest.approx(live.gamma)
    assert h["theta"] == pytest.approx(live.theta) and h["vega"] == pytest.approx(live.vega)
    assert h["greeks_forward"] == pytest.approx(live.forward)


# ── each failure reason, greeks left null ────────────────────────────────────────────────────────

@pytest.mark.parametrize("row, reason", [
    (_row(spot=None), "missing_spot"),
    (_row(symbol="B1VDA261120C00180000"), "nonstandard_contract"),       # adjusted root, not standard OSI
    (_row(symbol="NVDA260101C00180000"), "t_below_floor"),               # expiry before the session date
    (_row(symbol="NVDA261120C00100000", price=1.0, spot=181.0), "below_intrinsic"),
])
def test_each_failure_reason_stored_with_null_greeks(row: Dict[str, Any], reason: str) -> None:
    g = _price([row])[0]
    assert g["greeks_status"] == "failure" and g["greeks_reason"] == reason
    assert all(g[k] is None for k in ("iv", "delta", "gamma", "theta", "vega", "rho"))
    assert g["greeks_methodology"] == MV and g["greeks_basis"] == "trade"


# ── every non-greeks column copied exactly ───────────────────────────────────────────────────────

def test_every_non_greeks_column_is_copied_exactly() -> None:
    r = _row()
    g = _price([r])[0]
    for col in COLUMNS_NON_GREEKS:
        assert g[col] == r[col], col
    assert set(g.keys()) == set(COLUMNS)  # exactly the written columns, no more, no less


# ── the column contract against the real DDL (system of record, per greeks/SCHEMA.md) ────────────

def _canonical_ddl_path() -> Path:
    """The option_trade DDL path recorded in greeks/SCHEMA.md — the system of record is the Scala
    repo, not a copy here (SCHEMA.md: "Do not maintain a second copy of CREATE TABLE")."""
    schema_md = (Path(__file__).resolve().parents[2] / "greeks" / "SCHEMA.md").read_text(encoding="utf-8")
    m = re.search(r"Canonical DDL\s*\|\s*`([^`]+database_schema\.sql)`", schema_md)
    assert m is not None, "greeks/SCHEMA.md has no Canonical DDL path"
    return Path(m.group(1)).expanduser()


def _option_trade_columns(ddl: str) -> List[str]:
    """The column names in the trading.option_trade CREATE block."""
    start = ddl.find("CREATE TABLE IF NOT EXISTS trading.option_trade")
    assert start != -1, "trading.option_trade CREATE not found in the DDL"
    body = ddl[ddl.index("(", start) + 1:]
    cols: List[str] = []
    for line in body.splitlines():
        if line.lstrip().startswith(")"):
            break
        m = re.match(r"\s+(\w+)\s+\w", line)  # a column definition: name followed by a type token
        if m:
            cols.append(m.group(1))
    return cols


def test_writer_columns_match_the_option_trade_ddl() -> None:
    path = _canonical_ddl_path()
    if not path.exists():
        pytest.fail(f"option_trade DDL not found at {path} (system of record per greeks/SCHEMA.md)")
    ddl_cols = _option_trade_columns(path.read_text(encoding="utf-8"))
    # The writer writes every DDL column except ingested_at (the ReplacingMergeTree version DEFAULT).
    assert set(ddl_cols) - {"ingested_at"} == set(COLUMNS)
    assert "ingested_at" not in COLUMNS


# ── orchestration: ledger skip, dry run, CLI ─────────────────────────────────────────────────────

D1, D2 = date(2016, 1, 4), date(2016, 1, 5)


def _fake_deps(pairs: List[tuple[date, str]], ledgered: set[tuple[date, str]],
               record: Dict[str, List[Any]]) -> Deps:
    def process_pair(p: tuple[date, str]) -> PairResult:
        record["processed"].append(p)
        return PairResult(p[0], p[1], rows=2, priced=2, reasons={}, seconds=0.01, error=None)

    def count_rows(d: date, u: str) -> int:
        record["counted"].append((d, u))
        return 5

    return Deps(
        list_pairs=lambda a, b: list(pairs),
        ledgered=lambda a, b: set(ledgered),
        count_rows=count_rows,
        process_pair=process_pair,
        mapper=lambda fn, items: [fn(i) for i in items])  # serial


def test_a_rerun_skips_pairs_that_already_have_a_ledger_row() -> None:
    rec: Dict[str, List[Any]] = {"processed": [], "counted": []}
    deps = _fake_deps([(D1, "AAA"), (D1, "BBB"), (D2, "AAA")], {(D1, "AAA")}, rec)
    summary = run(D1, D2, dry_run=False, deps=deps)
    assert rec["processed"] == [(D1, "BBB"), (D2, "AAA")]  # (D1, AAA) already ledgered -> skipped
    assert summary.pairs == 2 and summary.rows == 4


def test_a_dry_run_writes_nothing_but_counts_the_pairs() -> None:
    rec: Dict[str, List[Any]] = {"processed": [], "counted": []}
    deps = _fake_deps([(D1, "AAA"), (D1, "BBB")], set(), rec)
    run(D1, D2, dry_run=True, deps=deps)
    assert rec["processed"] == []                      # nothing priced, nothing written
    assert set(rec["counted"]) == {(D1, "AAA"), (D1, "BBB")}


def test_entry_point_main_path_parses_and_runs() -> None:
    seen: Dict[str, Any] = {}

    def fake_build(processes: int) -> Deps:
        seen["processes"] = processes
        return _fake_deps([], set(), {"processed": [], "counted": []})

    def fake_run(a: date, b: date, dry: bool, deps: Deps):  # type: ignore[no-untyped-def]
        seen["args"] = (a, b, dry)
        return None

    main(["2016-01-04", "2016-01-06", "--processes", "3", "--dry-run"],
         build_deps=fake_build, run_fn=fake_run)
    assert seen["processes"] == 3
    assert seen["args"] == (date(2016, 1, 4), date(2016, 1, 6), True)


def test_processes_is_required_no_default() -> None:
    with pytest.raises(SystemExit):
        main(["2016-01-04", "2016-01-06"])  # argparse: --processes required


def test_dataset_name() -> None:
    assert dataset_for("v2") == "greeks_v2"


# ── the REAL entry point spawns forkserver workers that import the task by module path ───────────

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_entry_point_starts_forkserver_workers_as_a_subprocess() -> None:
    # `python -m greeks.history … --processes 2` with the self-test deps: real argparse → real
    # forkserver ProcessPoolExecutor → workers import greeks.history.worker._selftest_pair by module
    # path and run. Before the fix (worker in __main__) every worker raised AttributeError. No CH.
    env = {**os.environ, "GREEKS_HISTORY_SELFTEST": "ok"}
    proc = subprocess.run(
        [sys.executable, "-m", "greeks.history", "2016-01-04", "2016-01-05", "--processes", "2"],
        cwd=str(_REPO_ROOT), env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


def test_a_dying_worker_stops_the_run_with_an_error() -> None:
    # A worker that _exit()s mid-task is a dead worker, not a per-pair failure: ProcessPoolExecutor
    # raises BrokenProcessPool and the mapper stops the run at once with a specific error.
    from greeks.history.worker import _suicide_pair, pool_mapper
    mapper = pool_mapper(2, initializer=None)  # no ClickHouse setup; the suicide pair just exits
    with pytest.raises(RuntimeError, match="worker process died"):
        list(mapper(_suicide_pair, [(D1, "AAA"), (D2, "BBB")]))  # the mapper streams, so iterate it


# ── per-process setup built once, inputs cached per day ──────────────────────────────────────────

class _FakeRes:
    column_names: List[str] = []
    result_rows: List[Any] = []


class _FakeClient:
    """A ClickHouse client whose reads return no rows (so pricing is a no-op) and whose writes are
    swallowed — enough to drive _process_pair without a database."""
    def query(self, *a: Any, **k: Any) -> _FakeRes:
        return _FakeRes()

    def insert(self, *a: Any, **k: Any) -> None:
        return None


class _Cfg:
    methodology_version = "v2"
    t_floor_minutes = 15.0
    expiry_time_et = "16:00"


def _patch_worker(monkeypatch: pytest.MonkeyPatch, loader_days: List[date]) -> Dict[str, int]:
    """Replace the three per-process builders with counting fakes and reset the worker global."""
    import greeks.history.worker as w
    counts = {"config": 0, "client": 0, "loader": 0}

    def make_config() -> Any:
        counts["config"] += 1
        return _Cfg()

    def make_client() -> Any:
        counts["client"] += 1
        return _FakeClient()

    def make_loader(client: Any) -> Any:
        counts["loader"] += 1
        def load(day: date) -> Any:
            loader_days.append(day)
            return object()
        return load

    monkeypatch.setattr(w, "_WORKER", None)
    monkeypatch.setattr(w, "_make_config", make_config)
    monkeypatch.setattr(w, "_client", make_client)
    monkeypatch.setattr(w, "_make_loader", make_loader)
    return counts


def test_config_client_loader_built_once_per_process_not_per_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    import greeks.history.worker as w
    days: List[date] = []
    counts = _patch_worker(monkeypatch, days)
    w._worker_init()  # the pool initializer: runs once when the worker starts
    for pair in [(D1, "A"), (D1, "B"), (D2, "A")]:
        assert w._process_pair(pair).error is None
    assert counts == {"config": 1, "client": 1, "loader": 1}  # once per process, not per pair


def test_inputs_load_once_per_day_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    import greeks.history.worker as w
    days: List[date] = []
    _patch_worker(monkeypatch, days)
    w._worker_init()
    for pair in [(D1, "A"), (D1, "B"), (D1, "C")]:
        w._process_pair(pair)
    assert days == [D1]  # one load for the day, reused for every pair of it


def test_a_day_change_reloads_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    import greeks.history.worker as w
    days: List[date] = []
    _patch_worker(monkeypatch, days)
    w._worker_init()
    for pair in [(D1, "A"), (D1, "B"), (D2, "A"), (D2, "B"), (D1, "C")]:
        w._process_pair(pair)
    assert days == [D1, D2, D1]  # reload on each change; only the most recent day is cached


def test_day_line_has_the_new_format_with_failed_as_a_number(caplog: pytest.LogCaptureFixture) -> None:
    # One ok pair and one failed pair on D1: "1 underlying(s), 2 rows, 2 priced, 1 failed, 0.5s".
    def mapper(fn: Any, items: List[tuple[date, str]]) -> Any:
        yield PairResult(D1, "A", rows=2, priced=2, reasons={}, seconds=0.3, error=None)
        yield PairResult(D1, "B", rows=0, priced=0, reasons={}, seconds=0.2, error="boom")

    deps = Deps(list_pairs=lambda a, b: [(D1, "A"), (D1, "B")], ledgered=lambda a, b: set(),
                count_rows=lambda d, u: 0,
                process_pair=lambda p: PairResult(p[0], p[1], 0, 0, {}, 0.0, None),
                mapper=mapper)
    with caplog.at_level(logging.INFO, logger="greeks.history"):
        run(D1, D1, dry_run=False, deps=deps)
    line = next(m for m in (r.getMessage() for r in caplog.records if r.name == "greeks.history")
                if " underlying(s), " in m)
    assert line == f"[greeks] {D1}: 1 underlying(s), 2 rows, 2 priced, 1 failed, 0.5s"
    assert "failed=" not in line  # the old per-reason breakdown is gone


# ── one ledger read, and a day line as each day finishes ─────────────────────────────────────────

def _streaming_deps(pairs: List[tuple[date, str]], complete_order: List[tuple[date, str]], *,
                    rows: int = 2, priced: int = 2, seconds: float = 0.5,
                    ledger_calls: Dict[str, int] | None = None) -> Deps:
    """Deps whose mapper yields results in `complete_order` (possibly out of order), counting ledger
    reads."""
    def ledgered(a: date, b: date) -> set[tuple[date, str]]:
        if ledger_calls is not None:
            ledger_calls["n"] = ledger_calls.get("n", 0) + 1
        return set()

    def process_pair(p: tuple[date, str]) -> PairResult:
        return PairResult(p[0], p[1], rows=rows, priced=priced, reasons={}, seconds=seconds, error=None)

    def mapper(fn: Any, items: List[tuple[date, str]]) -> Any:
        for p in complete_order:
            yield fn(p)

    return Deps(list_pairs=lambda a, b: list(pairs), ledgered=ledgered, count_rows=lambda d, u: 0,
                process_pair=process_pair, mapper=mapper)


def test_ledgered_is_read_exactly_once_per_run() -> None:
    calls: Dict[str, int] = {}
    pairs = [(D1, "A"), (D1, "B"), (D2, "X")]
    deps = _streaming_deps(pairs, pairs, ledger_calls=calls)
    run(D1, D2, dry_run=False, deps=deps)
    assert calls["n"] == 1  # one ClickHouse read, not one per pair


def test_each_day_line_is_emitted_when_its_last_pair_completes_out_of_order(
        caplog: pytest.LogCaptureFixture) -> None:
    pairs = [(D1, "A"), (D1, "B"), (D2, "X")]
    # D2's only pair finishes FIRST, then D1's two — so D2's line must precede D1's (completion order,
    # not sorted), and both must precede the summary.
    deps = _streaming_deps(pairs, [(D2, "X"), (D1, "A"), (D1, "B")])
    with caplog.at_level(logging.INFO, logger="greeks.history"):
        run(D1, D2, dry_run=False, deps=deps)
    msgs = [r.getMessage() for r in caplog.records if r.name == "greeks.history"]

    def idx(prefix: str) -> int:
        return next(i for i, m in enumerate(msgs) if m.startswith(prefix))

    assert idx(f"[greeks] {D2}:") < idx(f"[greeks] {D1}:") < idx("[greeks] done:")


def test_summary_totals_match_the_day_lines(caplog: pytest.LogCaptureFixture) -> None:
    pairs = [(D1, "A"), (D1, "B"), (D2, "X")]  # rows/priced 2 each -> D1=4, D2=2
    deps = _streaming_deps(pairs, [(D1, "A"), (D2, "X"), (D1, "B")], rows=2, priced=2)
    with caplog.at_level(logging.INFO, logger="greeks.history"):
        summary = run(D1, D2, dry_run=False, deps=deps)
    day_lines = [r.getMessage() for r in caplog.records
                 if r.name == "greeks.history" and " underlying(s), " in r.getMessage()]
    day_rows = day_priced = 0
    for line in day_lines:
        m = re.search(r"(\d+) rows, (\d+) priced", line)
        assert m is not None
        day_rows += int(m.group(1))
        day_priced += int(m.group(2))
    assert summary.rows == day_rows == 6
    assert summary.priced == day_priced == 6
    assert summary.days == 2 and summary.pairs == 3
