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
from castra.metrics import MetricsRecord, load_metrics
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

    click.echo(click.style(f"\n[OK] experiment {name!r} created", fg="green"))


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

    click.echo(click.style(f"\n[OK] experiment {name!r} forked from {source!r}", fg="green"))


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
    click.echo(click.style("[OK] repair complete", fg="green"))


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


# --- Phase 8: finalize / ship / archive / promote / compare -----------------


def _resolve_worktree_or_die(name: str, project_root: Path) -> "worktree.WorktreeInfo":
    """Find the worktree for `name`, raise click.ClickException if missing."""
    w = worktree.info(name, project_root=project_root)
    if w is None:
        raise click.ClickException(f"no experiment named {name!r}")
    return w


def _load_spec_or_die(name: str, w: "worktree.WorktreeInfo") -> ExperimentSpec:
    config_path = w.path / "experiments" / name / "config.yaml"
    if not config_path.exists():
        raise click.ClickException(
            f"experiment {name!r} has no config.yaml at {config_path}; "
            "is the worktree corrupted?"
        )
    return ExperimentSpec.from_yaml(config_path)


def cmd_exp_finalize(name: str, gen: int, reason: str, *,
                     force: bool = False) -> None:
    """Mark an experiment as canonically concluded at generation `gen`.

    Writes `concluded_gen`, `concluded_reason`, `concluded_at` into the
    spec. By default refuses to overwrite an existing finalization;
    pass --force to override.
    """
    project_root = paths.project_root()
    w = _resolve_worktree_or_die(name, project_root)
    if not w.path.exists():
        raise click.ClickException(
            f"experiment {name!r} is archived; finalize must run before "
            "archive (or restore the worktree first)."
        )
    spec = _load_spec_or_die(name, w)
    if spec.concluded_gen is not None and not force:
        raise click.ClickException(
            f"experiment {name!r} is already finalized "
            f"(gen={spec.concluded_gen}, reason={spec.concluded_reason!r}). "
            "Pass --force to overwrite."
        )
    spec.concluded_gen = gen
    spec.concluded_reason = reason
    spec.concluded_at = _now_iso()
    config_path = w.path / "experiments" / name / "config.yaml"
    spec.to_yaml(config_path)
    click.echo(click.style(
        f"[OK] experiment {name!r} finalized at gen={gen}", fg="green"))
    click.echo(f"  reason:       {reason}")
    click.echo(f"  concluded_at: {spec.concluded_at}")
    click.echo(f"  config:       {config_path}")


def _is_under(child: Path, parent: Path) -> bool:
    """True if `child` is `parent` or contained within it."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def cmd_exp_ship(name: str, *, force: bool = False) -> None:
    """Copy a finalized experiment's `config.yaml` and `benchmarks/` from
    the worktree back to the main repo's `experiments/<name>/`.

    Requires finalize first (the spec must record concluded_gen). Code
    merging happens separately via git merge — `ship` only handles the
    artifact sync.
    """
    project_root = paths.project_root()
    w = _resolve_worktree_or_die(name, project_root)
    if not w.path.exists():
        raise click.ClickException(
            f"worktree for {name!r} is archived; cannot ship from it."
        )
    spec = _load_spec_or_die(name, w)
    if spec.concluded_gen is None and not force:
        raise click.ClickException(
            f"experiment {name!r} not finalized yet. Run "
            f"`castra exp finalize {name} --gen N --reason ...` first, or "
            "pass --force to ship anyway."
        )

    src_dir = w.path / "experiments" / name
    dst_dir = project_root / "experiments" / name
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_config = src_dir / "config.yaml"
    dst_config = dst_dir / "config.yaml"
    shutil.copy2(src_config, dst_config)
    click.echo(f"  shipped:  {dst_config}")

    src_benchmarks = src_dir / "benchmarks"
    dst_benchmarks = dst_dir / "benchmarks"
    if src_benchmarks.exists():
        if dst_benchmarks.exists():
            shutil.rmtree(dst_benchmarks)
        shutil.copytree(src_benchmarks, dst_benchmarks)
        click.echo(f"  shipped:  {dst_benchmarks}/")
    else:
        click.echo("  (no benchmarks/ to ship)")

    click.echo(click.style(
        f"\n[OK] experiment {name!r} shipped to {project_root}", fg="green"))


def cmd_exp_archive(name: str, *, force: bool = False) -> None:
    """Remove the experiment's worktree and venv. The branch and any
    shipped artifacts are retained.

    By default requires the experiment to have been finalized (and ideally
    shipped). Pass --force to archive without those guards.
    """
    project_root = paths.project_root()
    w = worktree.info(name, project_root=project_root)
    if w is None:
        # No worktree present. If the branch still exists this is already
        # archived (idempotent no-op). If the branch is also missing the
        # experiment never existed.
        if branch_exists(paths.branch_for(name), cwd=project_root):
            click.echo(f"experiment {name!r} already archived.")
            return
        raise click.ClickException(f"no experiment named {name!r}")
    if not w.path.exists():
        click.echo(f"experiment {name!r} already archived.")
        return
    spec = _load_spec_or_die(name, w)
    if spec.concluded_gen is None and not force:
        raise click.ClickException(
            f"experiment {name!r} not finalized; archiving would lose "
            "in-progress state. Pass --force to archive anyway, or finalize "
            "first."
        )
    # Always pass force=True at the git layer: archive's contract is to
    # dispose the working-state directory, leaving only the branch and
    # any shipped artifacts. Uncommitted scaffold (config.yaml etc.) is
    # expected to be present and is intentionally not preserved here.
    worktree.remove(name, force=True, project_root=project_root)
    click.echo(click.style(
        f"[OK] worktree for {name!r} removed (branch retained)", fg="green"))


def cmd_exp_promote(name: str, gen: int | None = None) -> None:
    """Copy a finalized checkpoint to `<project>/frontier/` and append to
    `FRONTIER.md`.

    `gen` defaults to `spec.concluded_gen`. The checkpoint is expected at
    `<worktree>/experiments/<name>/checkpoints/gen-<gen>.pt`; if missing,
    falls back to `latest.pt`. Promoted file lands at
    `<project>/frontier/<name>-gen<N>.pt`.
    """
    project_root = paths.project_root()
    w = _resolve_worktree_or_die(name, project_root)
    if not w.path.exists():
        raise click.ClickException(
            f"experiment {name!r} is archived; cannot promote."
        )
    spec = _load_spec_or_die(name, w)
    if spec.concluded_gen is None:
        raise click.ClickException(
            f"experiment {name!r} not finalized; finalize before promoting."
        )
    use_gen = spec.concluded_gen if gen is None else gen

    src_named = w.path / "experiments" / name / "checkpoints" / f"gen-{use_gen}.pt"
    src_latest = w.path / "experiments" / name / "checkpoints" / "latest.pt"
    if src_named.exists():
        src = src_named
    elif src_latest.exists():
        src = src_latest
        click.echo(f"  (gen-{use_gen}.pt missing; falling back to latest.pt)")
    else:
        raise click.ClickException(
            f"no checkpoint at {src_named} or {src_latest}; "
            "promote requires a saved checkpoint."
        )

    frontier_dir = project_root / "frontier"
    frontier_dir.mkdir(parents=True, exist_ok=True)
    dst = frontier_dir / f"{name}-gen{use_gen}.pt"
    shutil.copy2(src, dst)
    click.echo(f"  copied:    {src} -> {dst}")

    frontier_md = project_root / "FRONTIER.md"
    line = (f"- {_now_iso()}: promoted **{name}** gen={use_gen} "
            f"(reason: {spec.concluded_reason}) -> `{dst.relative_to(project_root)}`")
    if frontier_md.exists():
        existing = frontier_md.read_text()
        if not existing.endswith("\n"):
            existing += "\n"
        frontier_md.write_text(existing + line + "\n")
    else:
        frontier_md.write_text(
            "# Frontier\n\nPromoted checkpoints in chronological order.\n\n"
            + line + "\n"
        )
    click.echo(f"  appended:  {frontier_md}")
    click.echo(click.style(
        f"\n[OK] experiment {name!r} gen={use_gen} promoted", fg="green"))


def _resolve_experiment_dir(
    name: str, project_root: Path,
) -> tuple[Path, str] | None:
    """Return `(experiment_dir, status)` for `name`.

    Prefers the live worktree's `experiments/<name>/` if present; falls back
    to the shipped copy at `<project_root>/experiments/<name>/`. Returns
    None if neither exists.

    `status` is "active" / "finalized" (worktree present) or "shipped"
    (worktree gone, archived data in main repo).
    """
    w = worktree.info(name, project_root=project_root)
    if w is not None and w.path.exists():
        exp_dir = w.path / "experiments" / name
        if exp_dir.exists():
            return exp_dir, _experiment_status(w, project_root)

    shipped_dir = project_root / "experiments" / name
    if shipped_dir.exists():
        return shipped_dir, "shipped"
    return None


def _format_metric_value(v: float) -> str:
    """Compact numeric formatting: 4 significant figures, integer-shaped
    values without a decimal point."""
    if v == int(v) and abs(v) < 1e9:
        return f"{int(v)}"
    return f"{v:.4g}"


def _format_delta(d: float) -> str:
    if d == 0:
        return "="
    formatted = _format_metric_value(abs(d))
    return ("+" if d > 0 else "-") + formatted


def _format_metadata_value(v: Any) -> str:
    """Compact one-line rendering of metadata values. Multi-line values
    (e.g. `notes:`) get newlines collapsed to ' | ' so they don't blow
    out the side-by-side table."""
    if v is None:
        return "-"
    s = str(v).rstrip()
    if "\n" in s:
        s = " | ".join(line.strip() for line in s.splitlines() if line.strip())
    if len(s) > 80:
        s = s[:77] + "..."
    return s


def cmd_exp_compare(names: list[str]) -> None:
    """Print a side-by-side comparison of multiple experiments.

    Top section: spec metadata (status, concluded_gen / reason, git_sha).
    Bottom section: numeric metrics from `benchmarks/metrics.yaml` plus
    free-form metadata, with a `delta` column when exactly two experiments
    are compared (last - first).
    """
    if len(names) < 2:
        raise click.ClickException("`compare` needs at least two experiment names.")
    project_root = paths.project_root()

    rows: list[dict[str, Any]] = []
    metrics_per_name: dict[str, MetricsRecord | None] = {}

    for name in names:
        resolved = _resolve_experiment_dir(name, project_root)
        if resolved is None:
            click.echo(f"[WARN] no experiment named {name!r}; skipping.")
            continue
        exp_dir, status = resolved

        spec: ExperimentSpec | None = None
        config_path = exp_dir / "config.yaml"
        if config_path.exists():
            try:
                spec = ExperimentSpec.from_yaml(config_path)
            except Exception:
                pass

        metrics: MetricsRecord | None = None
        try:
            metrics = load_metrics(exp_dir)
        except Exception as e:
            click.echo(f"[WARN] {name}: metrics.yaml unreadable: {e}")

        rows.append({
            "name": name,
            "status": status,
            "concluded_gen": spec.concluded_gen if spec else None,
            "concluded_reason": spec.concluded_reason if spec else None,
            "axes": sorted(spec.axes.keys()) if spec else [],
            "git_sha": (spec.git_sha[:8] if (spec and spec.git_sha) else "?"),
        })
        metrics_per_name[name] = metrics

    if not rows:
        raise click.ClickException("no comparable experiments found.")

    # --- Section 1: spec metadata ---

    click.echo()
    click.echo("=== experiments ===")
    headers = ["name", "status", "concluded_gen", "concluded_reason", "axes", "git_sha"]
    widths = {h: max(len(h), max(len(str(r.get(h, "") or "")) for r in rows))
              for h in headers}
    sep = "  ".join("-" * widths[h] for h in headers)
    click.echo("  ".join(h.ljust(widths[h]) for h in headers))
    click.echo(sep)
    for r in rows:
        click.echo("  ".join(str(r.get(h, "") or "").ljust(widths[h]) for h in headers))

    # --- Section 2: metrics (only if at least one experiment has them) ---

    have_metrics = any(m is not None for m in metrics_per_name.values())
    if not have_metrics:
        return

    present_names = [r["name"] for r in rows]
    show_delta = len(present_names) == 2

    # Collect all numeric metric keys (preserve insertion order across experiments).
    metric_keys: list[str] = []
    seen: set[str] = set()
    for n in present_names:
        m = metrics_per_name.get(n)
        if m is None:
            continue
        for k in m.metrics.keys():
            if k not in seen:
                seen.add(k)
                metric_keys.append(k)

    # Same for metadata.
    meta_keys: list[str] = []
    seen_meta: set[str] = set()
    for n in present_names:
        m = metrics_per_name.get(n)
        if m is None:
            continue
        for k in m.metadata.keys():
            if k not in seen_meta:
                seen_meta.add(k)
                meta_keys.append(k)

    # Build value columns.
    def _val_for(name: str, key: str, kind: str) -> str:
        m = metrics_per_name.get(name)
        if m is None:
            return "-"
        if kind == "metric":
            v = m.metrics.get(key)
            return _format_metric_value(v) if v is not None else "-"
        if key not in m.metadata:
            return "-"
        return _format_metadata_value(m.metadata[key])

    def _delta_for(key: str) -> str:
        if not show_delta:
            return ""
        first, last = present_names[0], present_names[-1]
        m_first = metrics_per_name.get(first)
        m_last = metrics_per_name.get(last)
        if m_first is None or m_last is None:
            return "-"
        v_first = m_first.metrics.get(key)
        v_last = m_last.metrics.get(key)
        if v_first is None or v_last is None:
            return "-"
        return _format_delta(v_last - v_first)

    # Print metrics table.
    if metric_keys:
        click.echo()
        click.echo("=== metrics ===")
        m_headers = ["metric"] + present_names + (["diff"] if show_delta else [])
        m_rows: list[list[str]] = []
        for k in metric_keys:
            row = [k] + [_val_for(n, k, "metric") for n in present_names]
            if show_delta:
                row.append(_delta_for(k))
            m_rows.append(row)

        m_widths = [
            max(len(m_headers[i]), *(len(r[i]) for r in m_rows))
            for i in range(len(m_headers))
        ]
        click.echo("  ".join(h.ljust(m_widths[i]) for i, h in enumerate(m_headers)))
        click.echo("  ".join("-" * w for w in m_widths))
        for r in m_rows:
            click.echo("  ".join(c.ljust(m_widths[i]) for i, c in enumerate(r)))

    if meta_keys:
        click.echo()
        click.echo("=== metadata ===")
        md_headers = ["key"] + present_names
        md_rows: list[list[str]] = []
        for k in meta_keys:
            md_rows.append([k] + [_val_for(n, k, "metadata") for n in present_names])

        md_widths = [
            max(len(md_headers[i]), *(len(r[i]) for r in md_rows))
            for i in range(len(md_headers))
        ]
        click.echo("  ".join(h.ljust(md_widths[i]) for i, h in enumerate(md_headers)))
        click.echo("  ".join("-" * w for w in md_widths))
        for r in md_rows:
            click.echo("  ".join(c.ljust(md_widths[i]) for i, c in enumerate(r)))
