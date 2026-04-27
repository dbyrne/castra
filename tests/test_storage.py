"""Tests for the Store (SQLite shard + worker state) and JsonlLog."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from castra.storage import JsonlLog, Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "runs.db")


def test_submit_and_get(store: Store) -> None:
    sid = store.submit_shard("exp-1", "MyShard", {"x": 1})
    rec = store.get_shard(sid)
    assert rec is not None
    assert rec.experiment == "exp-1"
    assert rec.type_name == "MyShard"
    assert rec.payload == {"x": 1}
    assert rec.status == "pending"
    assert rec.attempts == 0


def test_claim_returns_pending_shard(store: Store) -> None:
    sid = store.submit_shard("exp-1", "MyShard", {"x": 1})
    rec = store.claim_next("worker-A")
    assert rec is not None
    assert rec.shard_id == sid
    assert rec.status == "claimed"
    assert rec.worker_id == "worker-A"
    assert rec.attempts == 1


def test_claim_returns_none_when_empty(store: Store) -> None:
    assert store.claim_next("worker-A") is None


def test_claim_filters_by_experiment(store: Store) -> None:
    store.submit_shard("exp-1", "S", {})
    sid2 = store.submit_shard("exp-2", "S", {})
    rec = store.claim_next("worker-A", experiment="exp-2")
    assert rec is not None
    assert rec.shard_id == sid2


def test_complete_marks_done(store: Store) -> None:
    sid = store.submit_shard("exp-1", "S", {})
    store.claim_next("worker-A")
    store.complete_shard(sid, {"answer": 42})
    rec = store.get_shard(sid)
    assert rec is not None
    assert rec.status == "completed"
    assert rec.result == {"answer": 42}


def test_fail_requeues_by_default(store: Store) -> None:
    sid = store.submit_shard("exp-1", "S", {})
    store.claim_next("worker-A")
    store.fail_shard(sid, "boom")
    rec = store.get_shard(sid)
    assert rec is not None
    assert rec.status == "pending"
    assert rec.error_text == "boom"
    assert rec.worker_id is None


def test_fail_no_requeue_marks_failed(store: Store) -> None:
    sid = store.submit_shard("exp-1", "S", {})
    store.claim_next("worker-A")
    store.fail_shard(sid, "permanent", requeue=False)
    rec = store.get_shard(sid)
    assert rec is not None
    assert rec.status == "failed"


def test_lease_expiry_reclaims(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    """Expired leases return shards to the pending pool on the next claim."""
    sid = store.submit_shard("exp-1", "S", {})
    # Claim with a 0-second lease (expires immediately).
    store.claim_next("worker-A", lease_seconds=0)
    # Tick time forward minimally (sqlite timestamps are iso strings; 0 seconds
    # means the lease has already passed by the time reclaim_expired runs).
    time.sleep(0.01)
    rec = store.claim_next("worker-B")
    assert rec is not None
    assert rec.shard_id == sid
    assert rec.worker_id == "worker-B"
    assert rec.attempts == 2  # incremented again


def test_shard_counts(store: Store) -> None:
    s1 = store.submit_shard("exp-1", "S", {})
    s2 = store.submit_shard("exp-1", "S", {})
    s3 = store.submit_shard("exp-1", "S", {})
    store.claim_next("w")
    counts = store.shard_counts()
    assert counts["pending"] == 2
    assert counts["claimed"] == 1


def test_submit_many(store: Store) -> None:
    ids = store.submit_many("exp-1", [
        ("S", {"i": 0}), ("S", {"i": 1}), ("S", {"i": 2}),
    ])
    assert len(ids) == 3
    rows = store.list_shards(experiment="exp-1")
    assert len(rows) == 3
    assert sorted(r.payload["i"] for r in rows) == [0, 1, 2]


def test_register_worker_and_heartbeat(store: Store) -> None:
    wid = store.register_worker("local-subprocess", "myhost")
    assert wid
    assert store.heartbeat(wid)
    assert store.heartbeat("ghost") is False


def test_list_workers(store: Store) -> None:
    store.register_worker("local-subprocess", "h1")
    store.register_worker("ssh", "h2")
    ws = store.list_workers()
    assert len(ws) == 2
    backends = sorted(w.backend for w in ws)
    assert backends == ["local-subprocess", "ssh"]


def test_jsonl_log_roundtrip(tmp_path: Path) -> None:
    log = JsonlLog(tmp_path / "run.jsonl")
    log.append("turn_start", {"turn": 1})
    log.append("move", {"unit": 0, "to": 5})
    log.append("turn_end", {})
    entries = log.read_all()
    assert [e["event_type"] for e in entries] == ["turn_start", "move", "turn_end"]
    assert entries[0]["detail"] == {"turn": 1}
