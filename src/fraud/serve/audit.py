"""Prediction audit trail: every scored transaction is persisted as an event.

SQLite through the standard library — one file, one table, WAL mode, a lock
around writes so the FastAPI worker threads never interleave. What is stored is
exactly what an investigator would ask for: what was scored, when, by which
model, under which policy, with what score, and what was returned.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id        TEXT    NOT NULL,
    endpoint          TEXT    NOT NULL,
    scored_at         TEXT    NOT NULL,          -- UTC, ISO 8601
    transaction_id    INTEGER NOT NULL,
    model_version     TEXT    NOT NULL,
    fraud_probability REAL    NOT NULL,
    risk_level        TEXT    NOT NULL,
    action            TEXT    NOT NULL,
    policy            TEXT    NOT NULL,          -- rank | threshold
    block_threshold   REAL    NOT NULL,
    review_threshold  REAL,                      -- threshold policy
    review_budget     INTEGER,                   -- rank policy
    review_cutoff     REAL,                      -- lowest probability reviewed in the batch
    batch_size        INTEGER NOT NULL,
    latency_ms        REAL    NOT NULL           -- whole request, shared by its rows
);
CREATE INDEX IF NOT EXISTS ix_prediction_events_txn ON prediction_events (transaction_id);
CREATE INDEX IF NOT EXISTS ix_prediction_events_time ON prediction_events (scored_at);
"""

COLUMNS = (
    "request_id", "endpoint", "scored_at", "transaction_id", "model_version",
    "fraud_probability", "risk_level", "action", "policy", "block_threshold",
    "review_threshold", "review_budget", "review_cutoff", "batch_size", "latency_ms",
)  # fmt: skip


@dataclass(frozen=True)
class PredictionEvent:
    request_id: str
    endpoint: str
    scored_at: str
    transaction_id: int
    model_version: str
    fraud_probability: float
    risk_level: str
    action: str
    policy: str
    block_threshold: float
    review_threshold: float | None
    review_budget: int | None
    review_cutoff: float | None
    batch_size: int
    latency_ms: float


def new_request_id() -> str:
    return uuid.uuid4().hex


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class AuditLog:
    """Append-only prediction events in a SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def record(self, events: list[PredictionEvent]) -> int:
        if not events:
            return 0
        rows = [tuple(asdict(e)[c] for c in COLUMNS) for e in events]
        placeholders = ", ".join("?" for _ in COLUMNS)
        sql = f"INSERT INTO prediction_events ({', '.join(COLUMNS)}) VALUES ({placeholders})"
        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()
        return len(rows)

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM prediction_events").fetchone()
        return int(row[0])

    def recent(self, limit: int = 50, transaction_id: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT id, " + ", ".join(COLUMNS) + " FROM prediction_events"
        params: tuple[Any, ...] = ()
        if transaction_id is not None:
            sql += " WHERE transaction_id = ?"
            params = (transaction_id,)
        sql += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            cur = self._conn.execute(sql, (*params, int(limit)))
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
