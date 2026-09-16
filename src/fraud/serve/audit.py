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
    latency_ms        REAL    NOT NULL,          -- whole request, shared by its rows
    transaction_dt    INTEGER                    -- the transaction's own clock (TransactionDT)
);
CREATE INDEX IF NOT EXISTS ix_prediction_events_txn ON prediction_events (transaction_id);
CREATE INDEX IF NOT EXISTS ix_prediction_events_time ON prediction_events (scored_at);

-- the delayed label: what a scored transaction turned out to be, once that was known
-- (docs/feedback.md). One row per transaction; observed_dt is on the TransactionDT clock.
CREATE TABLE IF NOT EXISTS outcomes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id INTEGER NOT NULL UNIQUE,
    is_fraud       INTEGER NOT NULL,
    event_dt       INTEGER NOT NULL,             -- TransactionDT of the transaction
    observed_dt    INTEGER NOT NULL,             -- TransactionDT clock when the label became known
    recorded_at    TEXT    NOT NULL,             -- UTC, when it was written here
    source         TEXT    NOT NULL              -- e.g. simulated_feed, api
);

-- every run of the label feed: the clock it advanced to and what it appended
CREATE TABLE IF NOT EXISTS label_feed (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_dt    INTEGER NOT NULL,
    recorded_at TEXT    NOT NULL,
    n_new       INTEGER NOT NULL,
    n_total     INTEGER NOT NULL,
    source      TEXT    NOT NULL
);

-- every API call to a scoring endpoint, successful or not (monitoring: rate, latency, errors)
CREATE TABLE IF NOT EXISTS requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT    NOT NULL,
    endpoint    TEXT    NOT NULL,
    started_at  TEXT    NOT NULL,
    status_code INTEGER NOT NULL,
    latency_ms  REAL    NOT NULL,
    n_rows      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_requests_time ON requests (started_at);

-- a compact snapshot of each scored row's inputs (monitoring: data drift)
CREATE TABLE IF NOT EXISTS input_features (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id            TEXT    NOT NULL,
    transaction_id        INTEGER NOT NULL,
    scored_at             TEXT    NOT NULL,
    amount                REAL    NOT NULL,
    product               TEXT    NOT NULL,
    card4                 TEXT,
    card6                 TEXT,
    device_type           TEXT,
    has_identity          INTEGER NOT NULL,
    addr1_missing         INTEGER NOT NULL,
    p_email_present       INTEGER NOT NULL,
    r_email_present       INTEGER NOT NULL,
    n_missing_transaction INTEGER NOT NULL,
    n_missing_identity    INTEGER NOT NULL,
    hour                  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_input_features_time ON input_features (scored_at);
"""

INPUT_COLUMNS = (
    "request_id", "transaction_id", "scored_at", "amount", "product", "card4", "card6",
    "device_type", "has_identity", "addr1_missing", "p_email_present", "r_email_present",
    "n_missing_transaction", "n_missing_identity", "hour",
)  # fmt: skip

COLUMNS = (
    "request_id", "endpoint", "scored_at", "transaction_id", "model_version",
    "fraud_probability", "risk_level", "action", "policy", "block_threshold",
    "review_threshold", "review_budget", "review_cutoff", "batch_size", "latency_ms",
    "transaction_dt",
)  # fmt: skip

OUTCOME_COLUMNS = ("transaction_id", "is_fraud", "event_dt", "observed_dt", "recorded_at", "source")

# columns added after the first release; applied to an existing file at open time
_MIGRATIONS = (("prediction_events", "transaction_dt", "INTEGER"),)


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
    transaction_dt: int | None = None


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
        for table, column, sql_type in _MIGRATIONS:
            present = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in present:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
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

    def record_request(
        self, request_id: str, endpoint: str, started_at: str, status_code: int,
        latency_ms: float, n_rows: int,
    ) -> None:  # fmt: skip
        with self._lock:
            self._conn.execute(
                "INSERT INTO requests "
                "(request_id, endpoint, started_at, status_code, latency_ms, n_rows) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (request_id, endpoint, started_at, status_code, latency_ms, n_rows),
            )
            self._conn.commit()

    def record_inputs(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        placeholders = ", ".join("?" for _ in INPUT_COLUMNS)
        sql = f"INSERT INTO input_features ({', '.join(INPUT_COLUMNS)}) VALUES ({placeholders})"
        with self._lock:
            self._conn.executemany(sql, [tuple(r[c] for c in INPUT_COLUMNS) for r in rows])
            self._conn.commit()
        return len(rows)

    def record_outcomes(self, rows: list[dict[str, Any]]) -> int:
        """Append delayed labels; a transaction already labelled is left as first recorded."""
        if not rows:
            return 0
        placeholders = ", ".join("?" for _ in OUTCOME_COLUMNS)
        sql = (
            f"INSERT OR IGNORE INTO outcomes ({', '.join(OUTCOME_COLUMNS)}) VALUES ({placeholders})"
        )
        with self._lock:
            before = self._conn.total_changes
            self._conn.executemany(sql, [tuple(r[c] for c in OUTCOME_COLUMNS) for r in rows])
            self._conn.commit()
            return self._conn.total_changes - before

    def record_feed_run(self, as_of_dt: int, n_new: int, source: str) -> int:
        with self._lock:
            total = int(self._conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0])
            self._conn.execute(
                "INSERT INTO label_feed (as_of_dt, recorded_at, n_new, n_total, source) "
                "VALUES (?, ?, ?, ?, ?)",
                (int(as_of_dt), utc_now(), int(n_new), total, source),
            )
            self._conn.commit()
        return total

    def feed_clock(self) -> int | None:
        """The TransactionDT clock the label feed has advanced to (None: never run)."""
        with self._lock:
            row = self._conn.execute("SELECT MAX(as_of_dt) FROM label_feed").fetchone()
        return None if row[0] is None else int(row[0])

    def reviewed_transactions(self, day: int) -> set[int]:
        """Transactions of one TransactionDT day whose latest decision is 'review'.

        Latest, because a re-scored transaction is a re-decision, not another
        review; the rank policy charges the day's budget with these.
        """
        lo, hi = day * 86_400, (day + 1) * 86_400
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.transaction_id FROM prediction_events e JOIN ("
                "  SELECT transaction_id, MAX(id) AS id FROM prediction_events"
                "  WHERE transaction_dt >= ? AND transaction_dt < ? GROUP BY transaction_id"
                ") last ON last.id = e.id WHERE e.action = 'review'",
                (lo, hi),
            ).fetchall()
        return {int(r[0]) for r in rows}

    def frame(self, table: str, since: str | None = None, until: str | None = None) -> Any:
        """A pandas frame of one table within a time window (monitoring reads)."""
        import pandas as pd

        time_col = {"requests": "started_at", "outcomes": "recorded_at"}.get(table, "scored_at")
        clauses, params = [], []
        if since:
            clauses.append(f"{time_col} >= ?")
            params.append(since)
        if until:
            clauses.append(f"{time_col} < ?")
            params.append(until)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            return pd.read_sql_query(
                f"SELECT * FROM {table}{where}", self._conn, params=tuple(params)
            )

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
