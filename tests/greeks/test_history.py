"""greeks.history: the trade-basis pricer shares the live core, stores each failure reason with null
greeks, copies every non-greeks column exactly, matches the option_trade DDL column contract, skips
ledgered pairs, writes nothing on a dry run, and parses its CLI. No live ClickHouse, no processes.
"""

from __future__ import annotations

import re
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


# ── the column contract against the DDL ──────────────────────────────────────────────────────────

def _ddl_columns(ddl: str) -> List[str]:
    body = ddl.split("(", 1)[1]
    cols: List[str] = []
    for line in body.splitlines():
        if line.strip().startswith(")"):
            break
        m = re.match(r"\s+(\w+)\s+\w", line)  # a column definition: name followed by a type token
        if m:
            cols.append(m.group(1))
    return cols


def test_writer_columns_match_the_option_trade_ddl() -> None:
    ddl = (Path(__file__).parent / "fixtures" / "database_schema.sql").read_text(encoding="utf-8")
    ddl_cols = _ddl_columns(ddl)
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
