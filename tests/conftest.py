"""Shared pytest fixtures: a sandbox git repo we can safely create worktrees in."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True,
                   capture_output=True, text=True)


@pytest.fixture
def project_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh git repo with one initial commit. cwd is set inside it."""
    repo = tmp_path / "myproj"
    repo.mkdir()
    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Castra Test", cwd=repo)
    _git("config", "commit.gpgsign", "false", cwd=repo)
    (repo / "README.md").write_text("# myproj\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-m", "initial", cwd=repo)

    # cd into the repo so castra's path resolution works naturally.
    monkeypatch.chdir(repo)
    return repo


@pytest.fixture
def project_repo_with_pyproject(project_repo: Path) -> Path:
    """Like `project_repo` but with a minimal pyproject.toml so install_editable
    has something to chew on. The package itself is empty.
    """
    pyproject = """\
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "myproj"
version = "0.0.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
include = ["myproj*"]
"""
    (project_repo / "pyproject.toml").write_text(pyproject)
    pkg = project_repo / "myproj"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    _git("add", "-A", cwd=project_repo)
    _git("commit", "-m", "add pyproject", cwd=project_repo)
    return project_repo
