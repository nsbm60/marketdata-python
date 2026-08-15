"""SQLite work queue for (contract, date) pull jobs.

Process-safe via ``BEGIN IMMEDIATE`` on claim/update. Single-writer friendly;
multiple processes may claim distinct rows. Embryo of a production queue —
kept separable from pull HTTP logic.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from greeks.domain import WorkStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS work_items (
    contract     TEXT NOT NULL,
    work_date    TEXT NOT NULL,  -- ISO date
    status       TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (contract, work_date)
);
CREATE INDEX IF NOT EXISTS idx_work_status ON work_items(status, work_date);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WorkItem:
    contract: str
    work_date: date
    status: WorkStatus
    attempt_count: int
    last_error: Optional[str]
    updated_at: datetime


class WorkQueue:
    """File-backed SQLite queue. Path is the DB file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=60.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> WorkQueue:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def enqueue(
        self,
        contract: str,
        work_date: date,
        *,
        status: WorkStatus = WorkStatus.PENDING,
    ) -> None:
        """Insert or leave existing row unchanged (idempotent seed)."""
        now = _utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO work_items (contract, work_date, status, attempt_count, last_error, updated_at)
            VALUES (?, ?, ?, 0, NULL, ?)
            ON CONFLICT(contract, work_date) DO NOTHING
            """,
            (contract, work_date.isoformat(), status.value, now),
        )
        self._conn.commit()

    def enqueue_many(
        self, items: Iterator[tuple[str, date]] | list[tuple[str, date]]
    ) -> int:
        n = 0
        now = _utc_now_iso()
        for contract, work_date in items:
            cur = self._conn.execute(
                """
                INSERT INTO work_items (contract, work_date, status, attempt_count, last_error, updated_at)
                VALUES (?, ?, ?, 0, NULL, ?)
                ON CONFLICT(contract, work_date) DO NOTHING
                """,
                (contract, work_date.isoformat(), WorkStatus.PENDING.value, now),
            )
            n += cur.rowcount
        self._conn.commit()
        return n

    def skip(self, contract: str, work_date: date, reason: str) -> None:
        self._set_status(contract, work_date, WorkStatus.SKIPPED, error=reason)

    def mark_done(self, contract: str, work_date: date) -> None:
        self._set_status(contract, work_date, WorkStatus.DONE, error=None)

    def mark_failed(self, contract: str, work_date: date, error: str) -> None:
        self._set_status(contract, work_date, WorkStatus.FAILED, error=error)

    def _set_status(
        self,
        contract: str,
        work_date: date,
        status: WorkStatus,
        *,
        error: Optional[str],
    ) -> None:
        now = _utc_now_iso()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                """
                UPDATE work_items
                SET status = ?, last_error = ?, updated_at = ?
                WHERE contract = ? AND work_date = ?
                """,
                (status.value, error, now, contract, work_date.isoformat()),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def claim_next(
        self,
        *,
        statuses: tuple[WorkStatus, ...] = (WorkStatus.PENDING,),
    ) -> Optional[WorkItem]:
        """Atomically claim the next pending item (status → in_progress)."""
        wanted = tuple(s.value for s in statuses)
        placeholders = ",".join("?" * len(wanted))
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                f"""
                SELECT contract, work_date, status, attempt_count, last_error, updated_at
                FROM work_items
                WHERE status IN ({placeholders})
                ORDER BY work_date ASC, contract ASC
                LIMIT 1
                """,
                wanted,
            ).fetchone()
            if row is None:
                self._conn.commit()
                return None
            now = _utc_now_iso()
            self._conn.execute(
                """
                UPDATE work_items
                SET status = ?, attempt_count = attempt_count + 1, updated_at = ?, last_error = NULL
                WHERE contract = ? AND work_date = ?
                """,
                (
                    WorkStatus.IN_PROGRESS.value,
                    now,
                    row["contract"],
                    row["work_date"],
                ),
            )
            self._conn.commit()
            return WorkItem(
                contract=row["contract"],
                work_date=date.fromisoformat(row["work_date"]),
                status=WorkStatus.IN_PROGRESS,
                attempt_count=int(row["attempt_count"]) + 1,
                last_error=None,
                updated_at=datetime.fromisoformat(now),
            )
        except Exception:
            self._conn.rollback()
            raise

    def get(self, contract: str, work_date: date) -> Optional[WorkItem]:
        row = self._conn.execute(
            """
            SELECT contract, work_date, status, attempt_count, last_error, updated_at
            FROM work_items
            WHERE contract = ? AND work_date = ?
            """,
            (contract, work_date.isoformat()),
        ).fetchone()
        if row is None:
            return None
        return _row_to_item(row)

    def count_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM work_items GROUP BY status"
        ).fetchall()
        return {str(r["status"]): int(r["n"]) for r in rows}


def _row_to_item(row: sqlite3.Row) -> WorkItem:
    return WorkItem(
        contract=row["contract"],
        work_date=date.fromisoformat(row["work_date"]),
        status=WorkStatus(row["status"]),
        attempt_count=int(row["attempt_count"]),
        last_error=row["last_error"],
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
