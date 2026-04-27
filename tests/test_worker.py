"""Tests for the worker poll loop, exercised against a fake client.

The HTTP layer is exercised separately in test_coordinator.py via TestClient;
here we want to lock down the worker's *control flow* — claim/run/complete/
fail/heartbeat sequencing — without spinning up a real server.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from castra import protocol
from castra.worker import import_modules, run_forever, run_one


@pytest.fixture(autouse=True)
def _clean_registry():
    protocol.clear_registry()
    yield
    protocol.clear_registry()


@pytest.fixture(autouse=True)
def _register_test_units():
    """Register the WorkUnit types used by these tests.

    We re-register inside the fixture (not at module top level) so the
    autouse `_clean_registry` fixture doesn't blow them away before each test.
    """

    @protocol.register("Doubler")
    class Doubler:
        def __init__(self, x: int):
            self.x = x

        @classmethod
        def from_dict(cls, data):
            return cls(x=data["x"])

        def to_dict(self):
            return {"x": self.x}

        def run(self):
            return {"y": self.x * 2}

    @protocol.register("Boomer")
    class Boomer:
        def __init__(self, msg: str = "boom"):
            self.msg = msg

        @classmethod
        def from_dict(cls, data):
            return cls(msg=data.get("msg", "boom"))

        def to_dict(self):
            return {"msg": self.msg}

        def run(self):
            raise RuntimeError(self.msg)

    yield


class FakeClient:
    """Implements the subset of CoordinatorClient that the worker uses."""

    def __init__(self, shards: list[dict] | None = None):
        self.shards_to_serve = list(shards or [])
        self.completed: list[tuple[str, dict[str, Any]]] = []
        self.failed: list[tuple[str, str, bool]] = []
        self.heartbeats = 0
        self.registered = False

    def register_worker(self, *, backend: str,
                        hostname: str | None = None) -> str:
        self.registered = True
        self.backend = backend
        return "worker-test"

    def claim(self, worker_id: str, *, experiment: str | None = None,
              lease_seconds: int = 240) -> dict[str, Any] | None:
        if not self.shards_to_serve:
            return None
        return self.shards_to_serve.pop(0)

    def complete(self, shard_id: str, *, worker_id: str,
                 result: dict[str, Any]) -> None:
        self.completed.append((shard_id, result))

    def fail(self, shard_id: str, *, worker_id: str, error: str,
             requeue: bool = True) -> None:
        self.failed.append((shard_id, error, requeue))

    def heartbeat(self, worker_id: str) -> None:
        self.heartbeats += 1

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def _shard(shard_id: str, type_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "shard_id": shard_id,
        "type_name": type_name,
        "payload": payload,
        "lease_expires_at": "",
        "attempts": 1,
    }


def test_run_one_executes_and_completes() -> None:
    client = FakeClient([_shard("s1", "Doubler", {"x": 21})])
    sid = run_one(client, "worker-test")
    assert sid == "s1"
    assert client.completed == [("s1", {"y": 42})]
    assert client.failed == []


def test_run_one_returns_none_when_empty() -> None:
    client = FakeClient([])
    assert run_one(client, "worker-test") is None


def test_run_one_failure_calls_fail_with_traceback() -> None:
    client = FakeClient([_shard("s1", "Boomer", {"msg": "oops"})])
    sid = run_one(client, "worker-test")
    assert sid == "s1"
    assert client.completed == []
    assert len(client.failed) == 1
    failed_sid, err, requeue = client.failed[0]
    assert failed_sid == "s1"
    assert "oops" in err
    assert "RuntimeError" in err
    assert requeue is True


def test_run_one_unknown_type_fails() -> None:
    """A shard of unregistered type should be reported as failed (not crash)."""
    client = FakeClient([_shard("s1", "GhostType", {})])
    run_one(client, "worker-test")
    assert client.completed == []
    assert len(client.failed) == 1
    assert "GhostType" in client.failed[0][1]


def test_run_forever_drains_with_max_idle() -> None:
    shards = [_shard(f"s{i}", "Doubler", {"x": i}) for i in range(3)]
    fake = FakeClient(shards)

    with patch("castra.worker.CoordinatorClient", return_value=fake):
        n = run_forever(
            "http://ignored",
            backend="test",
            max_idle_iterations=2,
            idle_sleep_s=0.0,
        )
    assert n == 3
    assert len(fake.completed) == 3
    assert fake.registered is True


def test_run_forever_heartbeats_after_each_completion() -> None:
    shards = [_shard(f"s{i}", "Doubler", {"x": i}) for i in range(2)]
    fake = FakeClient(shards)

    with patch("castra.worker.CoordinatorClient", return_value=fake):
        run_forever(
            "http://ignored",
            backend="test",
            max_idle_iterations=1,
            idle_sleep_s=0.0,
        )
    # One heartbeat per completed shard.
    assert fake.heartbeats == 2


def test_run_forever_stops_via_stop_when() -> None:
    """The stop_when callback should terminate the loop cleanly."""
    shards = [_shard(f"s{i}", "Doubler", {"x": i}) for i in range(100)]
    fake = FakeClient(shards)

    counter = {"n": 0}

    def should_stop() -> bool:
        counter["n"] += 1
        return counter["n"] > 5

    with patch("castra.worker.CoordinatorClient", return_value=fake):
        n = run_forever(
            "http://ignored",
            backend="test",
            max_idle_iterations=1000,
            idle_sleep_s=0.0,
            stop_when=should_stop,
        )
    # We processed 5 shards before stop_when returned True.
    assert n == 5


def test_import_modules_resolves_real_module() -> None:
    """import_modules wraps importlib; confirm it accepts an importable path."""
    import_modules(["json"])  # stdlib; no side effects
