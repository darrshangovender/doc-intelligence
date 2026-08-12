"""SQLite-backed human-review queue.

Schema is single-table — keeps the operational story simple and the failure
mode obvious. Auto-approved extractions are NOT stored here; the queue is
exclusively for items requiring human triage.

Lifecycle: ``add`` → ``pending`` → ``approve`` / ``reject``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from doc_intelligence.extractors.base import ExtractionResult, ExtractionStatus


SCHEMA = """
CREATE TABLE IF NOT EXISTS review_items (
    id TEXT PRIMARY KEY,
    doc_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected')),
    data_json TEXT NOT NULL,
    confidence_json TEXT NOT NULL,
    source_text TEXT NOT NULL,
    errors_json TEXT NOT NULL,
    source_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    reviewer TEXT,
    review_notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_items(status);
CREATE INDEX IF NOT EXISTS idx_review_doc_type ON review_items(doc_type);
"""


@dataclass
class ReviewRecord:
    id: str
    doc_type: str
    status: str
    data: dict[str, Any]
    confidence: dict[str, float]
    source_text: str
    errors: list[str]
    source_path: str | None
    created_at: str
    updated_at: str
    reviewer: str | None = None
    review_notes: str | None = None

    @property
    def overall_confidence(self) -> float:
        return min(self.confidence.values()) if self.confidence else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewQueue:
    """SQLite-backed queue for low-confidence / failed extractions."""

    def __init__(self, db_path: str | Path = "review_queue.db") -> None:
        self.db_path = Path(db_path)
        self._ensure_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    # --------------- writes ---------------

    def add(self, result: ExtractionResult, *, source_path: str | None = None) -> str:
        """Insert an extraction result, returning the new queue ID.

        Raises if the result was auto-approved — auto-approved items have no
        business in the queue.
        """
        if result.status == ExtractionStatus.AUTO_APPROVED:
            raise ValueError(
                "Auto-approved extractions should not enter the review queue."
            )
        record_id = uuid.uuid4().hex[:12]
        now = _now()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO review_items (
                    id, doc_type, status, data_json, confidence_json,
                    source_text, errors_json, source_path, created_at, updated_at
                ) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    result.doc_type,
                    json.dumps(result.data, default=str),
                    json.dumps(result.confidence),
                    result.source_text,
                    json.dumps(result.errors),
                    source_path,
                    now,
                    now,
                ),
            )
        return record_id

    def approve(
        self,
        record_id: str,
        *,
        reviewer: str | None = None,
        notes: str | None = None,
        corrected_data: dict[str, Any] | None = None,
    ) -> None:
        with self._conn() as c:
            row = c.execute("SELECT id FROM review_items WHERE id = ?", (record_id,)).fetchone()
            if row is None:
                raise KeyError(f"No queue item: {record_id}")
            updates: list[str] = ["status = 'approved'", "updated_at = ?"]
            params: list[Any] = [_now()]
            if reviewer is not None:
                updates.append("reviewer = ?")
                params.append(reviewer)
            if notes is not None:
                updates.append("review_notes = ?")
                params.append(notes)
            if corrected_data is not None:
                updates.append("data_json = ?")
                params.append(json.dumps(corrected_data, default=str))
            params.append(record_id)
            c.execute(f"UPDATE review_items SET {', '.join(updates)} WHERE id = ?", params)

    def reject(
        self, record_id: str, *, reviewer: str | None = None, notes: str | None = None
    ) -> None:
        with self._conn() as c:
            row = c.execute("SELECT id FROM review_items WHERE id = ?", (record_id,)).fetchone()
            if row is None:
                raise KeyError(f"No queue item: {record_id}")
            c.execute(
                """UPDATE review_items
                   SET status = 'rejected', updated_at = ?, reviewer = ?, review_notes = ?
                   WHERE id = ?""",
                (_now(), reviewer, notes, record_id),
            )

    # --------------- reads ---------------

    def get(self, record_id: str) -> ReviewRecord | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM review_items WHERE id = ?", (record_id,)).fetchone()
        return _row_to_record(row) if row else None

    def list(
        self,
        *,
        status: str | None = "pending",
        doc_type: str | None = None,
        limit: int = 100,
    ) -> list[ReviewRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if doc_type is not None:
            clauses.append("doc_type = ?")
            params.append(doc_type)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM review_items{where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def pending_count(self, *, doc_type: str | None = None) -> int:
        return len(self.list(status="pending", doc_type=doc_type, limit=10_000))


def _row_to_record(row: sqlite3.Row) -> ReviewRecord:
    return ReviewRecord(
        id=row["id"],
        doc_type=row["doc_type"],
        status=row["status"],
        data=json.loads(row["data_json"]),
        confidence=json.loads(row["confidence_json"]),
        source_text=row["source_text"],
        errors=json.loads(row["errors_json"]),
        source_path=row["source_path"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        reviewer=row["reviewer"],
        review_notes=row["review_notes"],
    )
