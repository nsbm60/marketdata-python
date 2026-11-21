"""_load_earnings: a resolved history row contributes its reaction day; an unresolved row (reaction
day unknown) contributes BOTH the reported day and the next trading day, tagged unresolved. Driven by
a fake ClickHouse client — no server."""

from __future__ import annotations

import datetime as dt
from typing import Any, List, Optional, Tuple

from ml.shared.calendar import next_trading_day

from iv_event import (
    EARNINGS_CALENDAR_TABLE, EARNINGS_HISTORY, EARNINGS_HISTORY_TABLE, EARNINGS_HISTORY_UNRESOLVED,
)
from iv_event.event import _load_earnings


class _Res:
    def __init__(self, rows: List[Tuple[Any, ...]]) -> None:
        self.result_rows = rows


class _FakeCh:
    """Answers the two reads _load_earnings issues, keyed by the table named in the SQL."""

    def __init__(self, history_rows: List[Tuple[Any, ...]],
                 calendar_rows: Optional[List[Tuple[Any, ...]]] = None) -> None:
        self._history = history_rows
        self._calendar = calendar_rows or []

    def query(self, sql: str, parameters: Optional[dict] = None) -> _Res:
        if EARNINGS_HISTORY_TABLE in sql:
            return _Res(self._history)
        if EARNINGS_CALENDAR_TABLE in sql:
            return _Res(self._calendar)
        raise AssertionError(f"unexpected query: {sql}")


def test_resolved_row_contributes_its_reaction_day_only() -> None:
    today = dt.date(2024, 8, 10)
    ch = _FakeCh([("AAA", dt.date(2024, 2, 1), dt.date(2024, 2, 2))])  # reaction known
    out = _load_earnings(ch, {"AAA"}, today)
    assert out["AAA"] == [(dt.date(2024, 2, 2), EARNINGS_HISTORY)]


def test_unresolved_row_contributes_reported_day_and_next_trading_day() -> None:
    today = dt.date(2024, 8, 10)
    reported = dt.date(2024, 8, 5)                 # a Monday
    nxt = next_trading_day(reported)               # the Tuesday
    assert nxt == dt.date(2024, 8, 6)
    ch = _FakeCh([("AAA", reported, None)])        # reaction unknown
    out = _load_earnings(ch, {"AAA"}, today)
    assert set(out["AAA"]) == {
        (reported, EARNINGS_HISTORY_UNRESOLVED),
        (nxt, EARNINGS_HISTORY_UNRESOLVED),
    }


def test_unresolved_next_trading_day_after_today_is_dropped_but_reported_kept() -> None:
    today = dt.date(2024, 8, 6)                    # Tuesday
    reported = today                               # the next trading day (Wed 8/7) is in the future
    assert next_trading_day(reported) > today
    ch = _FakeCh([("AAA", reported, None)])
    out = _load_earnings(ch, {"AAA"}, today)
    assert out["AAA"] == [(reported, EARNINGS_HISTORY_UNRESOLVED)]
