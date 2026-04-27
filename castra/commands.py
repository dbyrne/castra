"""Subcommand implementations for `castra exp ...`.

Each function here is invoked by the matching click command in `cli.py`.
They return None and print user-facing output via `click.echo`.
"""

from __future__ import annotations

import datetime as _dt
import shutil
from importlib import resources
from pathlib import Path
from typing import Any

import click
import yaml

from castra import paths, venv, worktree
from castra.git import GitError, branch_exists, current_sha
from castra.spec import ExperimentSpec, apply_overrides


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _load_template(template: str) -> dict[str, Any]:
    """Load a built-in template by name. Returns the parsed YAML dict."""
    try:
        text = (resources.files("castra.templates") / f"{template}.yaml").read_text()
    except FileNotFoundError as e:
        raise click.ClickException(f"unknown template: {template!r}") from e
    return yaml.safe_load(text) or {}


def _write_spec(spec: ExperimentSpec, exp_dir: Path) -> Path:
    config_path = exp_dir / "config.yaml"
    spec.to_yaml(config_path)
    return config_path


def _scaffold_experiment_dir(worktree_path: Path, name: str) -> Path:
    """Create the standard subdirectories inside the worktree."""
    exp_dir = worktree_path / "experiments" / name
    for sub in ("checkpoints", "runs", "benchmarks", "logs"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
    return exp_dir


def cmd_exp_create(name: str, template: str, overrides: list[str]) -> None:
    """Create a new experiment with worktree, branch, venv, and config."""
    project_root = paths.project_root()

    if branch_exists(paths.branch_for(name), cwd=project_root):
        raise click.ClickException(
            f"branch {paths.branch_for(name)!r} already exists; pick another name "
            f"or use `castra exp fork`."
        )

    spec_dict = _load_template(template)
    spec_dict["name"] = name
    apply_overrides(spec_dict, overrides)
    spec_dict.setdefault("created_at", _now_iso())
    try:
        spec_dict["git_sha"] = current_sha(cwd=project_root)
    except GitError:
        spec_dict["git_sha"] = None
    spec = ExperimentSpec.from_dict(spec_dict)

    click.echo(f"creating worktree for {name!r}...")
    wt_path = worktree.create(name, base_branch="main", project_root=project_root)
    click.echo(f"  worktree: {wt_path}")
    click.echo(f"  branch:   {paths.branch_for(name)}")

    exp_dir = _scaffold_experiment_dir(wt_path, name)
    config_path = _write_spec(spec, exp_dir)
    click.echo(f"  config:   {config_path}")

    click.echo("setting up venv...")
    vp = venv.ensure_venv(wt_path)
    click.echo(f"  venv:     {vp}")

    if (wt_path / "pyproject.toml").exists():
        click.echo("installing project in editable mode...")
        venv.install_editable_project(wt_path)

    click.echo(click.style(f"\n✓ experiment {name!r} created", fg="green"))


def cmd_exp_fork(name: str, source: str, checkpoint: str) -> None:
    """Fork a new experiment from an existing one."""
    project_root = paths.project_root()

    if branch_exists(paths.branch_for(name), cwd=project_root):
        raise click.ClickException(f"branch experiment/{name!r} already exists")

    source_branch = paths.branch_for(source)
    base_branch = source_branch if branch_exists(source_branch, cwd=project_root) else "main"
    click.echo(f"forking {name!r} from {source!r} (base: {base_branch})")

    # Try to load parent spec so we can inherit metadata.
    parent_spec: ExperimentSpec | None = None
    parent_wt = paths.worktree_path(source, root=project_root)
    parent_config = parent_wt / "experiments" / source / "config.yaml"
    if parent_config.exists():
        parent_spec = ExperimentSpec.from_yaml(parent_config)

    # Build the new spec by inheriting from parent if we have it; else minimal.
    if parent_spec is not None:
        spec_dict = parent_spec.to_dict()
    else:
        spec_dict = _load_template("base")

    spec_dict["name"] = name
    spec_dict["parent"] = source
    spec_dict["parent_checkpoint"] = checkpoint
    spec_dict["created_at"] = _now_iso()
    spec_dict["concluded_gen"] = None
    spec_dict["concluded_reason"] = None
    spec_dict["concluded_at"] = None
    try:
        spec_dict["git_sha"] = current_sha(cwd=project_root)
    except GitError:
        spec_dict["git_sha"] = None
    spec = ExperimentSpec.from_dict(spec_dict)

    click.echo(f"creating worktree for {name!r}...")
    wt_path = worktree.create(name, base_branch=base_branch, project_root=project_root)
    click.echo(f"  worktree: {wt_path}")

    exp_dir = _scaffold_experiment_dir(wt_path, name)
    config_path = _write_spec(spec, exp_dir)
    click.echo(f"  config:   {config_path}")

    # Inherit checkpoint if parent has one available. (Phase 1 stub: we copy
    # whatever file matches the policy without the gen-aware semantics from
    # the design — those land when we have actual training in Phase 5.)
    if parent_wt.exists():
        parent_ckpt_dir = parent_wt / "experiments" / source / "checkpoints"
        latest = parent_ckpt_dir / "latest.pt"
        if latest.exists():
            dest = exp_dir / "checkpoints" / "latest.pt"
            shutil.copy(latest, dest)
            click.echo(f"  ckpt:     {dest} (from parent latest)")

    click.echo("setting up venv...")
    venv.ensure_venv(wt_path)
    if (wt_path / "pyproject.toml").exists():
        venv.install_editable_project(wt_path)

    click.echo(click.style(f"\n✓ experiment {name!r} forked from {source!r}", fg="green"))


def cmd_exp_repair(name: str, *, rebuild_install: bool = True) -> None:
    """Idempotent venv re-setup for an existing experiment."""
    project_root = paths.project_root()
    wt_info = worktree.info(name, project_root=project_root)
    if wt_info is None:
        raise click.ClickException(f"no worktree found for experiment {name!r}")
    if not wt_info.path.exists():
        raise click.ClickException(f"worktree path does not exist: {wt_info.path}")

    click.echo(f"repairing {name!r} at {wt_info.path}")
    vp = venv.ensure_venv(wt_info.path)
    click.echo(f"  venv ok:  {vp}")
    if rebuild_install and (wt_info.path / "pyproject.toml").exists():
        click.echo("  reinstalling project in editable mode...")
        venv.install_editable_project(wt_info.path)
    click.echo(click.style("✓ repair complete", fg="green"))


def cmd_exp_list() -> None:
    """List all known experiments."""
    project_root = paths.project_root()
    rows = worktree.list_all(project_root=project_root)
    if not rows:
        click.echo("no experiments yet. create one with `castra exp create <name>`.")
        return

    name_w = max(len(w.name) for w in rows)
    click.echo(f"  {'NAME':<{name_w}}  STATUS    BRANCH                 PATH")
    click.echo(f"  {'-'*name_w}  --------  ---------------------  ----")
    for w in rows:
        status = _experiment_status(w, project_root)
        click.echo(f"  {w.name:<{name_w}}  {status:<8}  {w.branch:<22} {w.path}")


def cmd_exp_info(name: str) -> None:
    """Show detailed info for one experiment."""
    project_root = paths.project_root()
    w = worktree.info(name, project_root=project_root)
    if w is None:
        raise click.ClickException(f"no experiment named {name!r}")

    config_path = w.path / "experiments" / name / "config.yaml"
    spec: ExperimentSpec | None = None
    if config_path.exists():
        spec = ExperimentSpec.from_yaml(config_path)

    click.echo(click.style(f"experiment: {name}", bold=True))
    click.echo(f"  path:       {w.path}")
    click.echo(f"  branch:     {w.branch}")
    click.echo(f"  HEAD:       {w.head_sha}")
    click.echo(f"  status:     {_experiment_status(w, project_root)}")
    click.echo(f"  config:     {config_path if config_path.exists() else '(missing)'}")
    if spec is None:
        return
    click.echo(f"  parent:     {spec.parent or '(none)'}")
    if spec.parent:
        click.echo(f"  parent_ckpt:{spec.parent_checkpoint or '(none)'}")
    click.echo(f"  created_at: {spec.created_at or '(unknown)'}")
    click.echo(f"  axes:       {sorted(spec.axes.keys()) or '(none)'}")
    if spec.concluded_gen is not None:
        click.echo(f"  concluded:  gen={spec.concluded_gen} ({spec.concluded_reason})")
        click.echo(f"  concluded_at: {spec.concluded_at}")


def _experiment_status(w: worktree.WorktreeInfo,
                       project_root: Path) -> str:
    """Human-readable status: active / finalized / archived."""
    if not w.path.exists():
        return "archived"
    config_path = w.path / "experiments" / w.name / "config.yaml"
    if not config_path.exists():
        return "active"
    try:
        spec = ExperimentSpec.from_yaml(config_path)
    except Exception:
        return "active"
    if spec.concluded_gen is not None:
        return "final"
    return "active"
