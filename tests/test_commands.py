"""End-to-end tests for `castra exp create / fork / repair / list / info`."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from castra import paths
from castra.cli import main
from castra.spec import ExperimentSpec


def _run(args: list[str]) -> tuple[int, str]:
    """Run the click CLI inside the current cwd; return (exit_code, output)."""
    runner = CliRunner()
    result = runner.invoke(main, args, catch_exceptions=False)
    return result.exit_code, result.output


def test_create_makes_worktree_and_config(project_repo: Path) -> None:
    code, out = _run(["exp", "create", "exp-1"])
    assert code == 0, out
    wt = project_repo.parent / "myproj-exp-1"
    assert wt.exists()
    config = wt / "experiments" / "exp-1" / "config.yaml"
    assert config.exists()
    spec = ExperimentSpec.from_yaml(config)
    assert spec.name == "exp-1"
    assert spec.parent is None
    assert spec.created_at is not None
    assert spec.git_sha is not None


def test_create_with_overrides(project_repo: Path) -> None:
    code, out = _run([
        "exp", "create", "exp-1",
        "-o", "axes.nn.hidden_dim=128",
        "-o", "env.num_players=4",
    ])
    assert code == 0, out
    config = project_repo.parent / "myproj-exp-1" / "experiments" / "exp-1" / "config.yaml"
    spec = ExperimentSpec.from_yaml(config)
    assert spec.axes == {"nn": {"hidden_dim": 128}}
    assert spec.env == {"num_players": 4}


def test_create_scaffolds_subdirs(project_repo: Path) -> None:
    code, _ = _run(["exp", "create", "exp-1"])
    assert code == 0
    base = project_repo.parent / "myproj-exp-1" / "experiments" / "exp-1"
    for sub in ("checkpoints", "runs", "benchmarks", "logs"):
        assert (base / sub).is_dir(), f"missing {sub}"


def test_create_refuses_duplicate(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    code, out = _run(["exp", "create", "exp-1"])
    assert code != 0
    assert "already exists" in out


def test_list_shows_created(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    _run(["exp", "create", "exp-2"])
    code, out = _run(["exp", "list"])
    assert code == 0
    assert "exp-1" in out
    assert "exp-2" in out


def test_list_empty(project_repo: Path) -> None:
    code, out = _run(["exp", "list"])
    assert code == 0
    assert "no experiments yet" in out


def test_info_shows_spec(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1", "-o", "env.num_players=4"])
    code, out = _run(["exp", "info", "exp-1"])
    assert code == 0
    assert "exp-1" in out
    assert "experiment/exp-1" in out


def test_info_unknown_fails(project_repo: Path) -> None:
    code, out = _run(["exp", "info", "ghost"])
    assert code != 0
    assert "no experiment" in out.lower()


def test_fork_inherits_axes(project_repo: Path) -> None:
    _run(["exp", "create", "parent", "-o", "axes.nn.hidden_dim=128"])
    code, out = _run(["exp", "fork", "child", "--from", "parent"])
    assert code == 0, out
    child_config = (
        project_repo.parent / "myproj-child" / "experiments" / "child" / "config.yaml"
    )
    spec = ExperimentSpec.from_yaml(child_config)
    assert spec.parent == "parent"
    assert spec.parent_checkpoint == "auto"
    assert spec.axes == {"nn": {"hidden_dim": 128}}


def test_fork_branches_off_parent(project_repo: Path) -> None:
    """When parent's experiment branch exists, child should branch off it,
    NOT off main — so parent's code changes are inherited.
    """
    import subprocess

    _run(["exp", "create", "parent"])
    parent_wt = project_repo.parent / "myproj-parent"
    # Make a unique commit on parent's branch.
    (parent_wt / "PARENT_MARKER.txt").write_text("hello")
    subprocess.run(["git", "add", "-A"], cwd=str(parent_wt), check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t.t", "-c", "user.name=t",
         "commit", "-m", "parent change"],
        cwd=str(parent_wt), check=True,
    )

    _run(["exp", "fork", "child", "--from", "parent"])
    child_wt = project_repo.parent / "myproj-child"
    # The marker file should be present in child since it forked from parent's branch.
    assert (child_wt / "PARENT_MARKER.txt").exists()


def test_fork_falls_back_to_main_when_parent_branch_missing(project_repo: Path) -> None:
    """If we never `created` the source, fork should still succeed off main."""
    code, out = _run(["exp", "fork", "child", "--from", "ghost"])
    assert code == 0, out
    config = project_repo.parent / "myproj-child" / "experiments" / "child" / "config.yaml"
    spec = ExperimentSpec.from_yaml(config)
    assert spec.parent == "ghost"


def test_fork_inherits_parent_checkpoint_file(project_repo: Path) -> None:
    """If parent has experiments/<name>/checkpoints/latest.pt, fork copies it."""
    _run(["exp", "create", "parent"])
    parent_ckpts = (
        project_repo.parent / "myproj-parent" / "experiments" / "parent" / "checkpoints"
    )
    parent_ckpts.mkdir(parents=True, exist_ok=True)
    (parent_ckpts / "latest.pt").write_bytes(b"fake-checkpoint-bytes")

    _run(["exp", "fork", "child", "--from", "parent"])
    child_ckpt = (
        project_repo.parent / "myproj-child" / "experiments" / "child"
        / "checkpoints" / "latest.pt"
    )
    assert child_ckpt.exists()
    assert child_ckpt.read_bytes() == b"fake-checkpoint-bytes"


def test_repair_succeeds_after_create(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    code, out = _run(["exp", "repair", "exp-1", "--no-rebuild"])
    assert code == 0, out


def test_repair_unknown_fails(project_repo: Path) -> None:
    code, out = _run(["exp", "repair", "ghost", "--no-rebuild"])
    assert code != 0
