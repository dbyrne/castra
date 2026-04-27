"""Thin wrapper around `git` subprocess invocations.

Centralizing here keeps every other module free of subprocess plumbing
and makes it trivial to mock in tests if needed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    """A git command exited non-zero."""


def git(*args: str, cwd: Path | str | None = None, check: bool = True) -> str:
    """Run a git command and return stdout, stripped.

    Raises GitError on non-zero exit (when check=True, the default).
    """
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} exited {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def is_git_repo(path: Path) -> bool:
    """True if `path` is inside a git repo."""
    try:
        git("rev-parse", "--git-dir", cwd=path)
        return True
    except GitError:
        return False


def repo_root(start: Path | None = None) -> Path:
    """Return the absolute path to the top of the working tree containing `start`."""
    out = git("rev-parse", "--show-toplevel", cwd=start)
    return Path(out)


def current_branch(cwd: Path | None = None) -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)


def current_sha(cwd: Path | None = None) -> str:
    return git("rev-parse", "HEAD", cwd=cwd)


def branch_exists(name: str, cwd: Path | None = None) -> bool:
    """True if a local branch with this name exists."""
    try:
        git("rev-parse", "--verify", "--quiet", f"refs/heads/{name}", cwd=cwd)
        return True
    except GitError:
        return False
