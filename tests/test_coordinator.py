"""Tests for the FastAPI coordinator app."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from castra.coordinator import make_app
from castra.storage import Store


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    store = Store(tmp_path / "runs.db")
    app = make_app(store, experiment_label="test-exp")
    return TestClient(app)


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["experiment_label"] == "test-exp"


def test_register_worker(client: TestClient) -> None:
    r = client.post("/workers/register", json={"backend": "test", "hostname": "h"})
    assert r.status_code == 200
    assert "worker_id" in r.json()


def test_heartbeat_404_for_unknown(client: TestClient) -> None:
    r = client.post("/workers/ghost/heartbeat")
    assert r.status_code == 404


def test_submit_and_claim_flow(client: TestClient) -> None:
    # Submit two shards
    r = client.post("/shards/submit", json={
        "experiment": "exp-1",
        "shards": [
            {"type_name": "MyShard", "payload": {"x": 1}},
            {"type_name": "MyShard", "payload": {"x": 2}},
        ],
    })
    assert r.status_code == 200
    ids = r.json()["shard_ids"]
    assert len(ids) == 2

    # Register a worker
    wid = client.post(
        "/workers/register", json={"backend": "test"},
    ).json()["worker_id"]

    # Claim — should get the first one
    r = client.post("/shards/claim", json={"worker_id": wid})
    assert r.status_code == 200
    body = r.json()
    assert body["shard_id"] == ids[0]
    assert body["payload"] == {"x": 1}
    assert body["attempts"] == 1

    # Complete it
    r = client.post(f"/shards/{ids[0]}/complete",
                    json={"worker_id": wid, "result": {"ok": True}})
    assert r.status_code == 200

    # Claim again — should get the second
    r = client.post("/shards/claim", json={"worker_id": wid})
    assert r.status_code == 200
    assert r.json()["shard_id"] == ids[1]


def test_claim_returns_204_when_empty(client: TestClient) -> None:
    wid = client.post("/workers/register", json={"backend": "t"}).json()["worker_id"]
    r = client.post("/shards/claim", json={"worker_id": wid})
    assert r.status_code == 204


def test_complete_404_for_unknown_shard(client: TestClient) -> None:
    r = client.post("/shards/ghost/complete",
                    json={"worker_id": "x", "result": {}})
    assert r.status_code == 404


def test_complete_409_if_not_claimed(client: TestClient) -> None:
    sid = client.post("/shards/submit", json={
        "experiment": "exp", "shards": [{"type_name": "S", "payload": {}}],
    }).json()["shard_ids"][0]
    r = client.post(f"/shards/{sid}/complete",
                    json={"worker_id": "x", "result": {}})
    assert r.status_code == 409


def test_fail_requeues(client: TestClient) -> None:
    sid = client.post("/shards/submit", json={
        "experiment": "exp", "shards": [{"type_name": "S", "payload": {}}],
    }).json()["shard_ids"][0]
    wid = client.post("/workers/register", json={"backend": "t"}).json()["worker_id"]
    client.post("/shards/claim", json={"worker_id": wid})
    r = client.post(f"/shards/{sid}/fail",
                    json={"worker_id": wid, "error": "boom"})
    assert r.status_code == 200
    # Should be claimable again
    r = client.post("/shards/claim", json={"worker_id": wid})
    assert r.status_code == 200
    assert r.json()["shard_id"] == sid


def test_experiment_status(client: TestClient) -> None:
    client.post("/shards/submit", json={
        "experiment": "exp-1",
        "shards": [{"type_name": "S", "payload": {}}] * 3,
    })
    client.post("/workers/register", json={"backend": "test"})
    r = client.get("/experiments/exp-1/status")
    assert r.status_code == 200
    body = r.json()
    assert body["experiment"] == "exp-1"
    assert body["shard_counts"]["pending"] == 3
    assert len(body["workers"]) == 1


def test_list_shards_filters(client: TestClient) -> None:
    client.post("/shards/submit", json={
        "experiment": "exp-1",
        "shards": [{"type_name": "S", "payload": {"i": 0}}],
    })
    client.post("/shards/submit", json={
        "experiment": "exp-2",
        "shards": [{"type_name": "S", "payload": {"i": 0}}],
    })
    r = client.get("/shards", params={"experiment": "exp-2"})
    assert r.status_code == 200
    assert len(r.json()["shards"]) == 1
    assert r.json()["shards"][0]["experiment"] == "exp-2"
