"""PR4: SQLite work queue claim/done/fail."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from greeks.domain import WorkStatus
from greeks.queue.work_queue import WorkQueue


def test_enqueue_claim_done(tmp_path: Path) -> None:
    db = tmp_path / "q.db"
    with WorkQueue(db) as q:
        q.enqueue("NVDA260527C00100000", date(2026, 5, 27))
        q.enqueue("NVDA260527P00100000", date(2026, 5, 27))
        item = q.claim_next()
        assert item is not None
        assert item.status is WorkStatus.IN_PROGRESS
        assert item.attempt_count == 1
        q.mark_done(item.contract, item.work_date)
        got = q.get(item.contract, item.work_date)
        assert got is not None
        assert got.status is WorkStatus.DONE

        item2 = q.claim_next()
        assert item2 is not None
        q.mark_failed(item2.contract, item2.work_date, "boom")
        got2 = q.get(item2.contract, item2.work_date)
        assert got2 is not None
        assert got2.status is WorkStatus.FAILED
        assert got2.last_error == "boom"

        assert q.claim_next() is None
        counts = q.count_by_status()
        assert counts.get("done") == 1
        assert counts.get("failed") == 1


def test_enqueue_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "q.db"
    with WorkQueue(db) as q:
        d = date(2026, 5, 27)
        assert q.enqueue_many([("A", d), ("A", d), ("B", d)]) == 2
        assert q.enqueue_many([("A", d)]) == 0


def test_skip_excluded(tmp_path: Path) -> None:
    db = tmp_path / "q.db"
    with WorkQueue(db) as q:
        q.enqueue("X", date(2026, 6, 8))
        q.skip("X", date(2026, 6, 8), "excluded_date")
        got = q.get("X", date(2026, 6, 8))
        assert got is not None
        assert got.status is WorkStatus.SKIPPED
