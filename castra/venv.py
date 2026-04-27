"""Idempotent venv setup for an experiment worktree.

Each worktree has its own .venv with `--system-site-packages` so heavy
dependencies (pytorch, etc.) don't get duplicated, but the worktree's
editable install of the user project takes precedence.

Phase 1 only creates the venv; pip-installing the user project happens
in `repair`. We deliberately don't try to be smart here — Phase 1 is
scaffolding, not full automation.
"""

from __future__ import annotations

import subprocess
import sys
import venv as stdlib_venv
from pathlib import Path

from castra.paths import venv_path


def is_windows() -> bool:
    return sys.platform == "win32"


def python_in_venv(venv_dir: Path) -> Path:
    """Path to the python executable inside a venv."""
    if is_windows():
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def pip_in_venv(venv_dir: Path) -> Path:
    if is_windows():
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"


def ensure_venv(worktree: Path, *, system_site_packages: bool = True) -> Path:
    """Create the venv if absent. Returns the venv path. Idempotent."""
    vp = venv_path(worktree)
    if vp.exists() and python_in_venv(vp).exists():
        return vp
    builder = stdlib_venv.EnvBuilder(
        system_site_packages=system_site_packages,
        with_pip=True,
        clear=False,
    )
    builder.create(str(vp))
    return vp


def install_editable_project(worktree: Path) -> bool:
    """If the worktree has a pyproject.toml, pip install -e . into the venv.

    Returns True if an install was attempted, False if there's nothing to install.
    """
    if not (worktree / "pyproject.toml").exists():
        return False
    pip = pip_in_venv(venv_path(worktree))
    if not pip.exists():
        raise FileNotFoundError(f"pip not found at {pip}")
    subprocess.run(
        [str(pip), "install", "--quiet", "-e", "."],
        cwd=str(worktree),
        check=True,
    )
    return True
