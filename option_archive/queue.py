"""SQLite work queue for the option archive (PR1).

One row per ``(contract, date)`` pull job. Process-safe via ``BEGIN IMMEDIATE`` on
every claim/mutation; each worker process opens its **own** connection (WAL) — the
shared-connection contention that plagued the prior SQLite use does not apply.

**Fencing.** A lease reclaim cannot prove the original worker is dead — a genuinely
slow task (vendor stall, huge contract-day) can outlive the lease with its worker
alive, so two workers may end up finishing the same task. The data survives that
(ReplacingMergeTree collapses the double insert), but the queue state would not:
worker A's ``mark_done``/``fail_vendor`` must not land on worker B's claim. So each
claim carries a fresh ``claim_id`` and every mutation is
``UPDATE … WHERE contract=? AND work_date=? AND claim_id=?``. Zero rows matched
means the claim was superseded — logged and dropped, never applied.

No ``asyncio`` / ``threading`` (process fleet only). ``now`` is injectable so lease
and backoff are testable without wall-clock.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional
from uuid import uuid4

from option_archive.domain import OsiSymbol, TaskFailure, TaskStatus, WorkTask

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS work_items (
    contract          TEXT NOT NULL,
    work_date         TEXT NOT NULL,   -- ISO date
    status            TEXT NOT NULL,
    attempt_count     INTEGER NOT NULL DEFAULT 0,
    claim_id          TEXT,            -- fencing token of the current claim
    claimed_at        TEXT,            -- ISO UTC datetime of the current claim
    retry_not_before  TEXT,            -- ISO UTC; PENDING not claimable before this
    batch_id          TEXT,
    last_error        TEXT,
    updated_at        TEXT NOT NULL,
    PRIMARY KEY (contract, work_date)
);
CREATE INDEX IF NOT EXISTS idx_work_claimable ON work_items(status, work_date);
"""

_TERMINAL = (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.SKIPPED)


def _utc(now: Optional[datetime]) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(timezone.utc)


class WorkQueue:
    """File-backed SQLite queue. One connection per process (WAL)."""

    def __init__(
        self,
        path: Path | str,
        *,
        lease: timedelta,
        max_attempts: int,
        backoff_base: timedelta,
    ) -> None:
        if lease <= timedelta(0):
            raise ValueError("lease must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if backoff_base <= timedelta(0):
            raise ValueError("backoff_base must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lease = lease
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._conn = sqlite3.connect(str(self.path), timeout=60.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "WorkQueue":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # -- seeding ---------------------------------------------------------------

    def enqueue_many(self, items: Iterable[tuple[OsiSymbol, date]]) -> int:
        """Seed ``(contract, date)`` jobs. Idempotent: an existing row is left
        untouched (``INSERT OR IGNORE`` on the natural key) — re-enqueuing a pair
        never resets its status or attempt_count. PR5 gap-detection and the
        new-name trigger rely on exactly this silent-ignore-existing semantics.
        Returns the number of *newly* inserted rows.
        """
        now = _utc(None).isoformat()
        n = 0
        for contract, work_date in items:
            cur = self._conn.execute(
                """
                INSERT OR IGNORE INTO work_items
                    (contract, work_date, status, attempt_count, updated_at)
                VALUES (?, ?, ?, 0, ?)
                """,
                (str(contract), work_date.isoformat(), TaskStatus.PENDING.value, now),
            )
            n += cur.rowcount
        self._conn.commit()
        return n

    # -- claiming --------------------------------------------------------------

    def claim_next(self, *, now: Optional[datetime] = None) -> Optional[WorkTask]:
        """Atomically claim the oldest claimable job, stamping a fresh fencing
        token. Claimable = a PENDING row whose backoff has elapsed, or an
        IN_PROGRESS row whose lease has expired (dead-worker recovery). A reclaim
        is **not** an attempt — ``attempt_count`` is untouched here.
        """
        ts = _utc(now)
        now_iso = ts.isoformat()
        lease_cutoff = (ts - self._lease).isoformat()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                """
                SELECT contract, work_date, attempt_count, batch_id
                FROM work_items
                WHERE (status = ? AND (retry_not_before IS NULL OR retry_not_before <= ?))
                   OR (status = ? AND claimed_at IS NOT NULL AND claimed_at <= ?)
                ORDER BY work_date ASC, contract ASC
                LIMIT 1
                """,
                (
                    TaskStatus.PENDING.value,
                    now_iso,
                    TaskStatus.IN_PROGRESS.value,
                    lease_cutoff,
                ),
            ).fetchone()
            if row is None:
                self._conn.commit()
                return None
            claim_id = str(uuid4())
            self._conn.execute(
                """
                UPDATE work_items
                SET status = ?, claim_id = ?, claimed_at = ?, updated_at = ?
                WHERE contract = ? AND work_date = ?
                """,
                (
                    TaskStatus.IN_PROGRESS.value,
                    claim_id,
                    now_iso,
                    now_iso,
                    row["contract"],
                    row["work_date"],
                ),
            )
            self._conn.commit()
            return WorkTask(
                contract=OsiSymbol(str(row["contract"])),
                work_date=date.fromisoformat(row["work_date"]),
                status=TaskStatus.IN_PROGRESS,
                attempt_count=int(row["attempt_count"]),
                claim_id=claim_id,
                batch_id=row["batch_id"],
                claimed_at=ts,
                last_error=None,
            )
        except Exception:
            self._conn.rollback()
            raise

    # -- verdicts (all fenced on claim_id) -------------------------------------

    def mark_done(self, task: WorkTask, *, now: Optional[datetime] = None) -> None:
        """Terminal success. Call ONLY after the ClickHouse insert has committed."""
        self._fenced_update(
            task,
            "SET status = ?, updated_at = ?",
            (TaskStatus.DONE.value, _utc(now).isoformat()),
            verb="mark_done",
        )

    def fail_vendor(
        self,
        task: WorkTask,
        reason: TaskFailure,
        detail: str,
        *,
        now: Optional[datetime] = None,
    ) -> None:
        """A real vendor failure: +1 attempt, then back to PENDING with exponential
        backoff, or terminal FAILED (loud) once max_attempts is reached."""
        ts = _utc(now)
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cur = self._conn.execute(
                "SELECT attempt_count FROM work_items "
                "WHERE contract = ? AND work_date = ? AND claim_id = ?",
                (str(task.contract), task.work_date.isoformat(), self._require_claim(task)),
            ).fetchone()
            if cur is None:
                self._conn.commit()
                self._log_superseded("fail_vendor", task)
                return
            attempt = int(cur["attempt_count"]) + 1
            note = f"{reason.value}: {detail}"
            if attempt >= self._max_attempts:
                self._conn.execute(
                    """
                    UPDATE work_items
                    SET status = ?, attempt_count = ?, claim_id = NULL,
                        claimed_at = NULL, retry_not_before = NULL,
                        last_error = ?, updated_at = ?
                    WHERE contract = ? AND work_date = ? AND claim_id = ?
                    """,
                    (
                        TaskStatus.FAILED.value,
                        attempt,
                        note,
                        ts.isoformat(),
                        str(task.contract),
                        task.work_date.isoformat(),
                        task.claim_id,
                    ),
                )
                self._conn.commit()
                log.error(
                    "task %s %s FAILED terminally after %d attempts: %s",
                    task.contract,
                    task.work_date,
                    attempt,
                    note,
                )
                return
            backoff = self._backoff_base * (2 ** (attempt - 1))
            self._conn.execute(
                """
                UPDATE work_items
                SET status = ?, attempt_count = ?, claim_id = NULL, claimed_at = NULL,
                    retry_not_before = ?, last_error = ?, updated_at = ?
                WHERE contract = ? AND work_date = ? AND claim_id = ?
                """,
                (
                    TaskStatus.PENDING.value,
                    attempt,
                    (ts + backoff).isoformat(),
                    note,
                    ts.isoformat(),
                    str(task.contract),
                    task.work_date.isoformat(),
                    task.claim_id,
                ),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def release_transport(
        self, task: WorkTask, detail: str, *, now: Optional[datetime] = None
    ) -> None:
        """A connectivity/transport outage — NOT a vendor failure. Straight back to
        PENDING, ``attempt_count`` untouched, claim cleared, no backoff (the worker
        idle-polls until the link returns)."""
        self._fenced_update(
            task,
            "SET status = ?, claim_id = NULL, claimed_at = NULL, "
            "retry_not_before = NULL, last_error = ?, updated_at = ?",
            (TaskStatus.PENDING.value, detail, _utc(now).isoformat()),
            verb="release_transport",
        )

    # -- reporting -------------------------------------------------------------

    def counts(self) -> dict[TaskStatus, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM work_items GROUP BY status"
        ).fetchall()
        return {TaskStatus(r["status"]): int(r["n"]) for r in rows}

    def pending_exists(self) -> bool:
        """True while any row is still PENDING or IN_PROGRESS. A worker that gets
        ``None`` from ``claim_next`` sleeps while this is True (work is in backoff
        or leased) and exits once it is False (queue drained)."""
        row = self._conn.execute(
            "SELECT 1 FROM work_items WHERE status IN (?, ?) LIMIT 1",
            (TaskStatus.PENDING.value, TaskStatus.IN_PROGRESS.value),
        ).fetchone()
        return row is not None

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _require_claim(task: WorkTask) -> str:
        if task.claim_id is None:
            raise ValueError("mutation requires a claimed WorkTask (claim_id is None)")
        return task.claim_id

    def _fenced_update(
        self, task: WorkTask, set_clause: str, params: tuple[object, ...], *, verb: str
    ) -> None:
        claim_id = self._require_claim(task)
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cur = self._conn.execute(
                f"UPDATE work_items {set_clause} "
                "WHERE contract = ? AND work_date = ? AND claim_id = ?",
                (*params, str(task.contract), task.work_date.isoformat(), claim_id),
            )
            self._conn.commit()
            if cur.rowcount == 0:
                self._log_superseded(verb, task)
        except Exception:
            self._conn.rollback()
            raise

    @staticmethod
    def _log_superseded(verb: str, task: WorkTask) -> None:
        log.warning(
            "%s on %s %s ignored: claim %s was superseded by a lease reclaim",
            verb,
            task.contract,
            task.work_date,
            task.claim_id,
        )
