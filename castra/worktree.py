"""Git worktree lifecycle for castra-managed experiments.

Conventions:
- Each experiment gets its own worktree at `<project-name>-<exp-name>/`,
  sibling to the main repo, on a branch `experiment/<exp-name>`.
- Removing a worktree retains the branch (so the work history isn't lost).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from castra.git import branch_exists, git
from castra.paths import branch_for, worktree_path


@dataclass(frozen=True)
class WorktreeInfo:
    name: str          # experiment name (e.g. "exp-1")
    path: Path         # worktree filesystem path
    branch: str        # branch checked out in the worktree
    head_sha: str      # current HEAD


def create(name: str, base_branch: str = "main", *,
           project_root: Path | None = None) -> Path:
    """Create a new worktree at the conventional path.

    Branches off `base_branch`, creating a new branch `experiment/<name>`.
    Returns the worktree path.
    """
    path = worktree_path(name, root=project_root)
    if path.exists():
        raise FileExistsError(f"worktree path already exists: {path}")
    branch = branch_for(name)
    if branch_exists(branch, cwd=project_root):
        raise ValueError(
            f"branch {branch!r} already exists; use `fork` or pick a new name"
        )
    git("worktree", "add", str(path), "-b", branch, base_branch, cwd=project_root)
    return path


def remove(name: str, *, force: bool = False,
           project_root: Path | None = None) -> None:
    """Remove the worktree (branch is retained)."""
    path = worktree_path(name, root=project_root)
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(path))
    git(*args, cwd=project_root)


def list_all(project_root: Path | None = None) -> list[WorktreeInfo]:
    """Parse `git worktree list --porcelain` into structured records."""
    raw = git("worktree", "list", "--porcelain", cwd=project_root)
    if not raw:
        return []

    out: list[WorktreeInfo] = []
    cur: dict = {}
    for line in raw.splitlines() + [""]:  # blank sentinel flushes the last block
        if line == "":
            if cur:
                _maybe_append(cur, out)
                cur = {}
            continue
        if " " in line:
            key, _, value = line.partition(" ")
        else:
            key, value = line, ""
        cur[key] = value
    return out


def _maybe_append(block: dict, out: list[WorktreeInfo]) -> None:
    """Convert a parsed porcelain block into a WorktreeInfo, skipping the main repo."""
    path = Path(block.get("worktree", ""))
    if not path:
        return
    branch_ref = block.get("branch", "")
    branch = branch_ref.removeprefix("refs/heads/") if branch_ref else "(detached)"
    if not branch.startswith("experiment/"):
        # Only surface castra-managed worktrees in this list.
        return
    name = branch.removeprefix("experiment/")
    out.append(
        WorktreeInfo(
            name=name,
            path=path,
            branch=branch,
            head_sha=block.get("HEAD", ""),
        )
    )


def info(name: str, project_root: Path | None = None) -> WorktreeInfo | None:
    """Return info for a single named experiment's worktree, or None."""
    for w in list_all(project_root=project_root):
        if w.name == name:
            return w
    return None
