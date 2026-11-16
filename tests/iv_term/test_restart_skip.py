"""Restart-skip and ledger-coverage logic, from a fake ledger (no ClickHouse)."""

from __future__ import annotations

import datetime as dt

from iv_term import LEDGER_DATASET
from iv_term.term import ledger_rows, month_is_done

D1 = dt.date(2016, 1, 4)
D2 = dt.date(2016, 1, 5)


def test_month_is_done_when_every_present_pair_is_ledgered() -> None:
    present = {(D1, "AAPL"), (D1, "MSFT"), (D2, "AAPL")}
    ledgered = {(D1, "AAPL"), (D1, "MSFT"), (D2, "AAPL"), (D2, "NVDA")}
    assert month_is_done(present, ledgered) is True


def test_month_not_done_when_a_present_pair_is_missing_from_the_ledger() -> None:
    present = {(D1, "AAPL"), (D1, "MSFT")}
    ledgered = {(D1, "AAPL")}
    assert month_is_done(present, ledgered) is False


def test_empty_present_month_is_not_done() -> None:
    # No option trades that month: nothing to skip on (process_range is a cheap no-op either way).
    assert month_is_done(set(), {(D1, "AAPL")}) is False


def test_zero_row_pairs_are_still_ledgered_so_the_subset_holds_on_restart() -> None:
    # Correction A: a pair that produced NO iv_term rows is still ledgered (rows_inserted = 0), so a
    # fully-processed month is a subset of the ledger and is skipped on restart.
    present = {(D1, "AAPL"), (D1, "ZZZ")}  # ZZZ produced no qualifying groups
    per_pair = {(D1, "AAPL"): 3}           # only AAPL emitted rows
    rows = ledger_rows(present, per_pair)
    by_ticker = {(r[0], r[1]): r for r in rows}
    assert set(by_ticker) == present
    assert by_ticker[(D1, "AAPL")][2] == 3   # rows_inserted
    assert by_ticker[(D1, "ZZZ")][2] == 0
    # dataset / quote_failures / gap_reason / claimed_by columns match the ledger writer's shape.
    assert by_ticker[(D1, "ZZZ")][3] == 0
    assert by_ticker[(D1, "ZZZ")][4] == LEDGER_DATASET
    # And the ledger now covers the month, so a restart skips it.
    ledgered = {(r[0], r[1]) for r in rows}
    assert month_is_done(present, ledgered) is True


def test_ledger_rows_are_sorted_and_one_per_present_pair() -> None:
    present = {(D2, "AAPL"), (D1, "MSFT"), (D1, "AAPL")}
    rows = ledger_rows(present, {})
    assert [(r[0], r[1]) for r in rows] == [(D1, "AAPL"), (D1, "MSFT"), (D2, "AAPL")]
    assert all(r[2] == 0 for r in rows)
