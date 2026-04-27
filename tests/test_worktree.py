"""Worktree create/remove/list/info."""

from __future__ import annotations

from pathlib import Path

import pytest

from castra import paths, worktree
from castra.git import branch_exists


def test_create_makes_worktree(project_repo: Path) -> None:
    wt_path = worktree.create("exp-1", project_root=project_repo)
    assert wt_path.exists()
    assert wt_path == project_repo.parent / "myproj-exp-1"
    assert branch_exists("experiment/exp-1", cwd=project_repo)


def test_create_refuses_duplicate(project_repo: Path) -> None:
    worktree.create("exp-1", project_root=project_repo)
    with pytest.raises(Exception):
        worktree.create("exp-1", project_root=project_repo)


def test_list_returns_castra_worktrees_only(project_repo: Path) -> None:
    worktree.create("exp-1", project_root=project_repo)
    worktree.create("exp-2", project_root=project_repo)
    rows = worktree.list_all(project_root=project_repo)
    names = sorted(w.name for w in rows)
    assert names == ["exp-1", "exp-2"]


def test_list_skips_main_worktree(project_repo: Path) -> None:
    """The main repo's checkout should not appear in `list_all`."""
    worktree.create("exp-1", project_root=project_repo)
    rows = worktree.list_all(project_root=project_repo)
    assert all(w.name != "myproj" for w in rows)
    assert all(w.branch.startswith("experiment/") for w in rows)


def test_info_returns_record(project_repo: Path) -> None:
    worktree.create("exp-1", project_root=project_repo)
    info = worktree.info("exp-1", project_root=project_repo)
    assert info is not None
    assert info.name == "exp-1"
    assert info.branch == "experiment/exp-1"


def test_info_returns_none_for_unknown(project_repo: Path) -> None:
    assert worktree.info("ghost", project_root=project_repo) is None


def test_remove_keeps_branch(project_repo: Path) -> None:
    worktree.create("exp-1", project_root=project_repo)
    worktree.remove("exp-1", force=True, project_root=project_repo)
    assert branch_exists("experiment/exp-1", cwd=project_repo)


def test_paths_worktree_convention(project_repo: Path) -> None:
    p = paths.worktree_path("foo", root=project_repo)
    assert p == project_repo.parent / "myproj-foo"


def test_paths_branch_convention() -> None:
    assert paths.branch_for("foo") == "experiment/foo"
