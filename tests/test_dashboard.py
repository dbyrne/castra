"""Tests for the dashboard data layer + a smoke test on rendering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from castra.dashboard import (
    DashboardSnapshot,
    fetch_snapshot,
    render,
)


def test_fetch_snapshot_no_coordinator(project_repo: Path) -> None:
    """Without a coordinator URL, snapshot should still list experiments."""
    from castra import worktree
    worktree.create("exp-1", project_root=project_repo)
    snap = fetch_snapshot(coordinator_url=None, project_root=project_repo)
    assert any(e["name"] == "exp-1" for e in snap.experiments)
    assert snap.coordinator_url is None
    assert snap.healthz is None


def test_fetch_snapshot_marks_status(project_repo: Path) -> None:
    """Experiments with a config.yaml are 'active'; concluded ones are 'final'."""
    from castra import worktree
    from castra.spec import ExperimentSpec
    wt = worktree.create("exp-1", project_root=project_repo)
    config = wt / "experiments" / "exp-1" / "config.yaml"
    spec = ExperimentSpec(name="exp-1")
    spec.to_yaml(config)
    snap = fetch_snapshot(coordinator_url=None, project_root=project_repo)
    row = [e for e in snap.experiments if e["name"] == "exp-1"][0]
    assert row["status"] == "active"

    spec.concluded_gen = 50
    spec.concluded_reason = "best so far"
    spec.to_yaml(config)
    snap = fetch_snapshot(coordinator_url=None, project_root=project_repo)
    row = [e for e in snap.experiments if e["name"] == "exp-1"][0]
    assert row["status"] == "final"


def test_fetch_snapshot_handles_unreachable_coordinator(project_repo: Path) -> None:
    """When the coordinator URL doesn't resolve, error is captured."""
    snap = fetch_snapshot(
        coordinator_url="http://127.0.0.1:1",  # closed port
        project_root=project_repo,
    )
    assert snap.coordinator_error is not None
    assert "coordinator unreachable" in snap.coordinator_error


def test_fetch_snapshot_with_mocked_coordinator(project_repo: Path) -> None:
    fake_healthz = {
        "ok": True,
        "experiment_label": "exp-1",
        "shard_counts": {"pending": 5, "claimed": 2, "completed": 10},
    }
    fake_status = {
        "experiment": "exp-1",
        "shard_counts": fake_healthz["shard_counts"],
        "workers": [
            {"worker_id": "abc-12345", "backend": "local-subprocess",
             "hostname": "host-1", "last_heartbeat_at": "2026-04-27T00:00:00",
             "status": "active"},
        ],
    }

    with patch("castra.dashboard.CoordinatorClient") as MockCC:
        instance = MagicMock()
        instance.healthz = MagicMock(return_value=fake_healthz)
        instance.status = MagicMock(return_value=fake_status)
        instance.__enter__ = MagicMock(return_value=instance)
        instance.__exit__ = MagicMock(return_value=False)
        MockCC.return_value = instance
        snap = fetch_snapshot(
            coordinator_url="http://localhost:8765",
            project_root=project_repo,
        )
    assert snap.healthz == fake_healthz
    assert len(snap.workers) == 1
    assert snap.workers[0]["backend"] == "local-subprocess"


def test_render_does_not_crash_empty() -> None:
    """Smoke: render an empty snapshot."""
    snap = DashboardSnapshot()
    layout = render(snap)
    assert layout is not None  # got a Layout back


def test_render_does_not_crash_with_data() -> None:
    snap = DashboardSnapshot(
        coordinator_url="http://localhost:8765",
        experiments=[
            {"name": "exp-1", "status": "active", "branch": "experiment/exp-1",
             "path": "/x/y", "parent": None, "axes": ["nn"]},
        ],
        healthz={"experiment_label": "exp-1",
                 "shard_counts": {"pending": 1, "completed": 5}},
        workers=[{"worker_id": "abc", "backend": "test",
                  "hostname": "h", "last_heartbeat_at": "ts"}],
    )
    layout = render(snap)
    assert layout is not None


def test_render_handles_coordinator_error() -> None:
    snap = DashboardSnapshot(
        coordinator_url="http://x",
        coordinator_error="coordinator unreachable: connection refused",
    )
    layout = render(snap)
    assert layout is not None
