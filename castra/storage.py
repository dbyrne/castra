"""SQLite-backed coordinator state and JSONL run-event logs.

Two storage layers, each with a clear job:

- `Store`: durable shard + worker tables in SQLite. Used by the coordinator
  for the claim/complete/fail/heartbeat flow.
- `JsonlLog`: append-only event log for one run, used for replay.

The Store opens a fresh sqlite3 connection per call so it's safe to use from
multiple threads (including the request thread pool fastapi uses for sync
endpoints).
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS shards (
  shard_id TEXT PRIMARY KEY,
  experiment TEXT NOT NULL,
  type_name TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL,
  worker_id TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  submitted_at TEXT NOT NULL,
  claimed_at TEXT,
  lease_expires_at TEXT,
  completed_at TEXT,
  result_json TEXT,
  error_text TEXT
);
CREATE INDEX IF NOT EXISTS shards_status ON shards(status);
CREATE INDEX IF NOT EXISTS shards_experiment_status ON shards(experiment, status);

CREATE TABLE IF NOT EXISTS workers (
  worker_id TEXT PRIMARY KEY,
  backend TEXT NOT NULL,
  hostname TEXT,
  registered_at TEXT NOT NULL,
  last_heartbeat_at TEXT NOT NULL,
  status TEXT NOT NULL
);
"""


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


@dataclass
class ShardRecord:
    shard_id: str
    experiment: str
    type_name: str
    payload: dict[str, Any]
    status: str          # pending | claimed | completed | failed
    worker_id: str | None
    attempts: int
    submitted_at: str
    claimed_at: str | None
    lease_expires_at: str | None
    completed_at: str | None
    result: dict[str, Any] | None
    error_text: str | None


@dataclass
class WorkerRecord:
    worker_id: str
    backend: str
    hostname: str | None
    registered_at: str
    last_heartbeat_at: str
    status: str          # active | dead


class Store:
    """SQLite-backed coordinator state."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(str(self.db_path), timeout=30.0)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode = WAL")
        c.execute("PRAGMA synchronous = NORMAL")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def _init_schema(self) -> None:
        with self._init_lock, self._conn() as c:
            c.executescript(SCHEMA)

    # --- shards ---

    def submit_shard(self, experiment: str, type_name: str,
                     payload: dict[str, Any]) -> str:
        sid = str(uuid.uuid4())
        with self._conn() as c:
            c.execute(
                """INSERT INTO shards(shard_id, experiment, type_name,
                                      payload_json, status, submitted_at, attempts)
                   VALUES (?, ?, ?, ?, 'pending', ?, 0)""",
                (sid, experiment, type_name, json.dumps(payload), utc_now_iso()),
            )
        return sid

    def submit_many(self, experiment: str,
                    items: list[tuple[str, dict[str, Any]]]) -> list[str]:
        ids: list[str] = []
        now = utc_now_iso()
        with self._conn() as c:
            for type_name, payload in items:
                sid = str(uuid.uuid4())
                c.execute(
                    """INSERT INTO shards(shard_id, experiment, type_name,
                                          payload_json, status, submitted_at, attempts)
                       VALUES (?, ?, ?, ?, 'pending', ?, 0)""",
                    (sid, experiment, type_name, json.dumps(payload), now),
                )
                ids.append(sid)
        return ids

    def reclaim_expired(self) -> int:
        """Return shards whose lease expired to the pending pool. Returns count."""
        now_str = utc_now_iso()
        with self._conn() as c:
            cur = c.execute(
                """UPDATE shards
                   SET status = 'pending', worker_id = NULL,
                       claimed_at = NULL, lease_expires_at = NULL
                   WHERE status = 'claimed' AND lease_expires_at < ?""",
                (now_str,),
            )
            return cur.rowcount

    def claim_next(self, worker_id: str, *, experiment: str | None = None,
                   lease_seconds: int = 240) -> ShardRecord | None:
        """Atomically claim the oldest pending shard (optionally filtered to
        an experiment). Returns None if no pending shards.
        """
        self.reclaim_expired()
        now = _dt.datetime.now(_dt.timezone.utc)
        lease_until = (now + _dt.timedelta(seconds=lease_seconds)).isoformat()

        with self._conn() as c:
            where = "status = 'pending'"
            params: list[Any] = []
            if experiment is not None:
                where += " AND experiment = ?"
                params.append(experiment)

            row = c.execute(
                f"SELECT shard_id FROM shards WHERE {where} "
                f"ORDER BY submitted_at LIMIT 1",
                params,
            ).fetchone()
            if row is None:
                return None
            sid = row["shard_id"]

            # Conditional update — guards against losing a race with another
            # claimer. If status changed since SELECT, we'll see 0 rows and
            # return None.
            cur = c.execute(
                """UPDATE shards
                   SET status = 'claimed', worker_id = ?, claimed_at = ?,
                       lease_expires_at = ?, attempts = attempts + 1
                   WHERE shard_id = ? AND status = 'pending'""",
                (worker_id, now.isoformat(), lease_until, sid),
            )
            if cur.rowcount == 0:
                return None
            return self._row_to_shard(
                c.execute("SELECT * FROM shards WHERE shard_id = ?", (sid,)).fetchone()
            )

    def complete_shard(self, shard_id: str, result: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                """UPDATE shards
                   SET status = 'completed', completed_at = ?, result_json = ?
                   WHERE shard_id = ?""",
                (utc_now_iso(), json.dumps(result), shard_id),
            )

    def fail_shard(self, shard_id: str, error: str, *,
                   requeue: bool = True) -> None:
        """Mark a shard failed. If `requeue`, return it to pending; else
        leave it in 'failed' for human triage.
        """
        with self._conn() as c:
            if requeue:
                c.execute(
                    """UPDATE shards
                       SET status = 'pending', worker_id = NULL,
                           claimed_at = NULL, lease_expires_at = NULL,
                           error_text = ?
                       WHERE shard_id = ?""",
                    (error, shard_id),
                )
            else:
                c.execute(
                    """UPDATE shards SET status = 'failed', error_text = ?
                       WHERE shard_id = ?""",
                    (error, shard_id),
                )

    def get_shard(self, shard_id: str) -> ShardRecord | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM shards WHERE shard_id = ?", (shard_id,)
            ).fetchone()
        return self._row_to_shard(row) if row else None

    def list_shards(self, *, experiment: str | None = None,
                    status: str | None = None) -> list[ShardRecord]:
        sql = "SELECT * FROM shards"
        where: list[str] = []
        params: list[Any] = []
        if experiment is not None:
            where.append("experiment = ?")
            params.append(experiment)
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY submitted_at"
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [self._row_to_shard(r) for r in rows]

    def shard_counts(self, *, experiment: str | None = None) -> dict[str, int]:
        sql = "SELECT status, COUNT(*) AS c FROM shards"
        params: list[Any] = []
        if experiment is not None:
            sql += " WHERE experiment = ?"
            params.append(experiment)
        sql += " GROUP BY status"
        with self._conn() as c:
            return {r["status"]: r["c"]
                    for r in c.execute(sql, params).fetchall()}

    # --- workers ---

    def register_worker(self, backend: str, hostname: str | None) -> str:
        wid = str(uuid.uuid4())
        with self._conn() as c:
            c.execute(
                """INSERT INTO workers(worker_id, backend, hostname,
                                       registered_at, last_heartbeat_at, status)
                   VALUES (?, ?, ?, ?, ?, 'active')""",
                (wid, backend, hostname, utc_now_iso(), utc_now_iso()),
            )
        return wid

    def heartbeat(self, worker_id: str) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE workers SET last_heartbeat_at = ? WHERE worker_id = ?",
                (utc_now_iso(), worker_id),
            )
            return cur.rowcount > 0

    def list_workers(self) -> list[WorkerRecord]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM workers ORDER BY registered_at"
            ).fetchall()
        return [self._row_to_worker(r) for r in rows]

    # --- row mappers ---

    @staticmethod
    def _row_to_shard(row: sqlite3.Row) -> ShardRecord:
        return ShardRecord(
            shard_id=row["shard_id"],
            experiment=row["experiment"],
            type_name=row["type_name"],
            payload=json.loads(row["payload_json"]),
            status=row["status"],
            worker_id=row["worker_id"],
            attempts=row["attempts"],
            submitted_at=row["submitted_at"],
            claimed_at=row["claimed_at"],
            lease_expires_at=row["lease_expires_at"],
            completed_at=row["completed_at"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error_text=row["error_text"],
        )

    @staticmethod
    def _row_to_worker(row: sqlite3.Row) -> WorkerRecord:
        return WorkerRecord(
            worker_id=row["worker_id"],
            backend=row["backend"],
            hostname=row["hostname"],
            registered_at=row["registered_at"],
            last_heartbeat_at=row["last_heartbeat_at"],
            status=row["status"],
        )


class JsonlLog:
    """Append-only JSONL event log for one run.

    Each line: {"ts": iso8601, "event_type": str, "detail": {...}}.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event_type: str, detail: dict[str, Any]) -> None:
        line = json.dumps({
            "ts": utc_now_iso(),
            "event_type": event_type,
            "detail": detail,
        })
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
