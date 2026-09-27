"""WorkQueue behaviour, incl. the lease-reclaim fencing race (PR1)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from option_archive.domain import TaskFailure, TaskStatus, WorkTask, to_osi
from option_archive.queue import WorkQueue

_T0 = datetime(2026, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
_C1 = to_osi("NVDA260417C00180000")
_C2 = to_osi("NVDA260417P00180000")
_D = date(2026, 4, 1)


def _q(tmp_path: Path, **kw: object) -> WorkQueue:
    return WorkQueue(
        tmp_path / "q.db",
        lease=timedelta(minutes=kw.get("lease_min", 10)),  # type: ignore[arg-type]
        max_attempts=int(kw.get("max_attempts", 3)),  # type: ignore[arg-type]
        backoff_base=timedelta(seconds=int(kw.get("backoff_s", 60))),  # type: ignore[arg-type]
    )


def test_enqueue_is_idempotent(tmp_path: Path) -> None:
    q = _q(tmp_path)
    assert q.enqueue_many([(_C1, _D), (_C2, _D)]) == 2
    # re-enqueue leaves existing rows untouched; zero new rows
    assert q.enqueue_many([(_C1, _D), (_C2, _D)]) == 0
    assert q.counts()[TaskStatus.PENDING] == 2


def test_claim_oldest_first_and_marks_in_progress(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.enqueue_many([(_C1, date(2026, 4, 2)), (_C2, date(2026, 4, 1))])
    t = q.claim_next(now=_T0)
    assert t is not None
    assert t.work_date == date(2026, 4, 1)  # oldest work_date first
    assert t.status is TaskStatus.IN_PROGRESS
    assert t.claim_id is not None
    assert t.attempt_count == 0  # claiming is not an attempt


def test_mark_done_terminal(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.enqueue_many([(_C1, _D)])
    t = q.claim_next(now=_T0)
    assert t is not None
    q.mark_done(t, now=_T0)
    assert q.counts() == {TaskStatus.DONE: 1}
    assert not q.pending_exists()


def test_lease_reclaim_after_expiry(tmp_path: Path) -> None:
    q = _q(tmp_path, lease_min=10)
    q.enqueue_many([(_C1, _D)])
    first = q.claim_next(now=_T0)
    assert first is not None
    # before lease expiry: nothing claimable
    assert q.claim_next(now=_T0 + timedelta(minutes=5)) is None
    # after lease expiry: reclaimable, fresh token, still not an attempt
    second = q.claim_next(now=_T0 + timedelta(minutes=11))
    assert second is not None
    assert second.claim_id != first.claim_id
    assert second.attempt_count == 0


def test_fencing_stale_verdict_is_noop(tmp_path: Path) -> None:
    # The core race: worker A's claim is reclaimed by worker B; A's late verdict
    # must not land on B's row.
    q = _q(tmp_path, lease_min=10)
    q.enqueue_many([(_C1, _D)])
    a = q.claim_next(now=_T0)
    assert a is not None
    b = q.claim_next(now=_T0 + timedelta(minutes=11))  # B reclaims
    assert b is not None and b.claim_id != a.claim_id
    # A finally returns and tries to complete — must be a no-op
    q.mark_done(a, now=_T0 + timedelta(minutes=12))
    assert q.counts()[TaskStatus.IN_PROGRESS] == 1  # still B's claim, not DONE
    # B's verdict applies normally
    q.mark_done(b, now=_T0 + timedelta(minutes=13))
    assert q.counts() == {TaskStatus.DONE: 1}


def test_fail_vendor_backoff_then_terminal(tmp_path: Path) -> None:
    q = _q(tmp_path, max_attempts=2, backoff_s=60)
    q.enqueue_many([(_C1, _D)])

    t1 = q.claim_next(now=_T0)
    assert t1 is not None
    q.fail_vendor(t1, TaskFailure.VENDOR_ERROR, "boom", now=_T0)
    # back to pending but in backoff — not yet claimable
    assert q.claim_next(now=_T0 + timedelta(seconds=30)) is None
    # after backoff, claimable again with attempt_count carried
    t2 = q.claim_next(now=_T0 + timedelta(seconds=61))
    assert t2 is not None
    assert t2.attempt_count == 1

    q.fail_vendor(t2, TaskFailure.VENDOR_ERROR, "boom again", now=_T0 + timedelta(seconds=61))
    # hit max_attempts -> terminal FAILED, nothing left claimable
    assert q.counts()[TaskStatus.FAILED] == 1
    assert not q.pending_exists()


def test_release_transport_does_not_count_as_attempt(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.enqueue_many([(_C1, _D)])
    t = q.claim_next(now=_T0)
    assert t is not None
    q.release_transport(t, "connection reset", now=_T0)
    # immediately claimable again (no backoff), attempt_count still 0
    again = q.claim_next(now=_T0)
    assert again is not None
    assert again.attempt_count == 0


def test_pending_exists_distinguishes_drained_from_waiting(tmp_path: Path) -> None:
    q = _q(tmp_path, backoff_s=60)
    q.enqueue_many([(_C1, _D)])
    t = q.claim_next(now=_T0)
    assert t is not None
    q.fail_vendor(t, TaskFailure.VENDOR_ERROR, "x", now=_T0)
    # nothing claimable (in backoff) but the queue is NOT drained
    assert q.claim_next(now=_T0) is None
    assert q.pending_exists() is True


def test_mutation_requires_a_claim(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.enqueue_many([(_C1, _D)])
    unclaimed = WorkTask(
        contract=_C1, work_date=_D, status=TaskStatus.PENDING, attempt_count=0
    )  # claim_id is None — mutating it is a programming error, not a silent no-op
    with pytest.raises(ValueError):
        q.mark_done(unclaimed)
