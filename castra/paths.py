"""Path conventions for castra.

A castra-managed project lives in a normal git repo. Each experiment is a
git worktree placed as a sibling of the main repo:

    /code/myproj/                  <- main repo
    /code/myproj-experiment-1/     <- worktree on branch experiment/experiment-1
    /code/myproj-experiment-2/     <- worktree on branch experiment/experiment-2

Inside a worktree, the experiment's runtime state lives at
`experiments/<name>/` (config, checkpoints, runs, logs).

Inside the *main* repo, `experiments/<name>/` only contains state that has
been `ship`-ed back from the worktree (final config + benchmarks).
"""

from __future__ import annotations

from pathlib import Path

from castra.git import repo_root


def project_root(start: Path | None = None) -> Path:
    """Top of the main repo. Resolves through worktrees."""
    return repo_root(start)


def project_name(root: Path | None = None) -> str:
    return (root or project_root()).name


def worktree_path(name: str, root: Path | None = None) -> Path:
    """Convention: sibling of main, named `<project>-<name>`."""
    r = root or project_root()
    return r.parent / f"{r.name}-{name}"


def experiment_dir(name: str, *, in_worktree: bool, root: Path | None = None) -> Path:
    """Where an experiment's state lives.

    Inside its own worktree: `<worktree>/experiments/<name>/`.
    In the main repo (post-ship): `<main>/experiments/<name>/`.
    """
    if in_worktree:
        base = worktree_path(name, root=root)
    else:
        base = root or project_root()
    return base / "experiments" / name


def branch_for(name: str) -> str:
    """The branch convention for an experiment named `<name>`."""
    return f"experiment/{name}"


def venv_path(worktree: Path) -> Path:
    return worktree / ".venv"
