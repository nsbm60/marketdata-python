"""Restart-skip and ledger-coverage logic, from a fake ledger (no ClickHouse)."""

from __future__ import annotations

import datetime as dt

from iv_curve import LEDGER_DATASET
from iv_curve.curve import ledger_rows, month_is_done

D1 = dt.date(2016, 1, 4)
D2 = dt.date(2016, 1, 5)


def test_month_is_done_when_every_present_pair_is_ledgered() -> None:
    present = {(D1, "AAPL"), (D1, "MSFT"), (D2, "AAPL")}
    ledgered = {(D1, "AAPL"), (D1, "MSFT"), (D2, "AAPL"), (D2, "NVDA")}
    assert month_is_done(present, ledgered) is True


def test_month_not_done_when_a_present_pair_is_missing() -> None:
    assert month_is_done({(D1, "AAPL"), (D1, "MSFT")}, {(D1, "AAPL")}) is False


def test_empty_present_month_is_not_done() -> None:
    assert month_is_done(set(), {(D1, "AAPL")}) is False


def test_pairs_with_no_ok_expiry_are_still_ledgered_so_restart_skips() -> None:
    # A (day, underlying) present in iv_term_daily but with no ok expiry gets no curve rows, yet is
    # still ledgered (rows_inserted = 0) — so a fully-processed month is a subset of the ledger.
    present = {(D1, "AAPL"), (D1, "ZZZ")}   # ZZZ produced no curve rows
    per_pair = {(D1, "AAPL"): 7}            # AAPL wrote 7 rows (expiries + horizons)
    rows = ledger_rows(present, per_pair)
    by_ticker = {(r[0], r[1]): r for r in rows}
    assert set(by_ticker) == present
    assert by_ticker[(D1, "AAPL")][2] == 7
    assert by_ticker[(D1, "ZZZ")][2] == 0
    assert by_ticker[(D1, "ZZZ")][4] == LEDGER_DATASET
    assert month_is_done(present, {(r[0], r[1]) for r in rows}) is True


def test_ledger_rows_are_sorted_and_one_per_present_pair() -> None:
    present = {(D2, "AAPL"), (D1, "MSFT"), (D1, "AAPL")}
    rows = ledger_rows(present, {})
    assert [(r[0], r[1]) for r in rows] == [(D1, "AAPL"), (D1, "MSFT"), (D2, "AAPL")]
    assert all(r[2] == 0 for r in rows)
