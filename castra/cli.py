"""castra command-line entry point."""

from __future__ import annotations

from pathlib import Path

import click

from castra import commands


@click.group()
@click.version_option()
def main() -> None:
    """castra — experiment management and worker fleet for AI agent research."""


@main.group()
def exp() -> None:
    """Experiment lifecycle commands."""


@exp.command()
@click.argument("name")
@click.option("-t", "--template", default="base",
              help="Built-in template name (default: base).")
@click.option("-o", "--override", "overrides", multiple=True,
              help="Dotted-key override, e.g. -o axes.nn.training.lr=1e-4.")
def create(name: str, template: str, overrides: tuple[str, ...]) -> None:
    """Create a new experiment with worktree, branch, venv, and config."""
    commands.cmd_exp_create(name, template, list(overrides))


@exp.command()
@click.argument("name")
@click.option("--from", "source", required=True,
              help="Source experiment to fork from.")
@click.option("--checkpoint", default="auto",
              type=click.Choice(["auto", "latest", "finalized"], case_sensitive=False),
              help="Which parent checkpoint to inherit (default: auto).")
def fork(name: str, source: str, checkpoint: str) -> None:
    """Fork a new experiment from an existing one, inheriting code + checkpoint."""
    commands.cmd_exp_fork(name, source, checkpoint)


@exp.command()
@click.argument("name")
@click.option("--no-rebuild", "no_rebuild", is_flag=True,
              help="Skip the editable-install step.")
def repair(name: str, no_rebuild: bool) -> None:
    """Re-run venv setup idempotently for an existing experiment."""
    commands.cmd_exp_repair(name, rebuild_install=not no_rebuild)


@exp.command(name="list")
def exp_list() -> None:
    """List all experiments."""
    commands.cmd_exp_list()


@exp.command()
@click.argument("name")
def info(name: str) -> None:
    """Show detailed information about an experiment."""
    commands.cmd_exp_info(name)


@exp.command()
@click.option("--coordinator", "coord_url", default=None,
              help="Coordinator URL to query for live shard/worker state.")
@click.option("--refresh-hz", default=1.0, show_default=True,
              help="Refresh frequency in Hz.")
def dashboard(coord_url: str | None, refresh_hz: float) -> None:
    """Live TUI showing experiments + coordinator + workers (Ctrl+C to exit)."""
    from castra.dashboard import run as run_dashboard
    run_dashboard(coord_url, refresh_hz=refresh_hz)


@main.group()
def coordinator() -> None:
    """Coordinator process — fastapi sidecar owning shard state."""


@coordinator.command(name="start")
@click.argument("experiment")
@click.option("--port", default=8765, help="HTTP port to bind (default 8765).")
@click.option("--host", default="127.0.0.1",
              help="Host to bind (default 127.0.0.1; use 0.0.0.0 for LAN).")
def coordinator_start(experiment: str, port: int, host: str) -> None:
    """Start the coordinator for `experiment` (foreground, blocking)."""
    from castra import paths
    from castra.coordinator import serve
    project_root = paths.project_root()
    wt = paths.worktree_path(experiment, root=project_root)
    if not wt.exists():
        raise click.ClickException(
            f"no worktree for experiment {experiment!r}; "
            f"create one with `castra exp create {experiment}` first."
        )
    db_path = wt / "experiments" / experiment / "runs.db"
    click.echo(f"coordinator: experiment {experiment!r}")
    click.echo(f"  db:    {db_path}")
    click.echo(f"  bind:  http://{host}:{port}")
    serve(db_path, host=host, port=port, experiment_label=experiment)


@main.command()
@click.option("--coordinator", "coord_url", required=True,
              help="Coordinator URL, e.g. http://localhost:8765.")
@click.option("--experiment", default=None,
              help="Restrict to shards from this experiment.")
@click.option("--import", "imports", multiple=True,
              help="Module to import before polling (registers WorkUnit types). Repeatable.")
@click.option("--backend", default="local-subprocess",
              help="Backend label recorded with the worker registration.")
@click.option("--lease-seconds", default=240, show_default=True)
@click.option("--max-idle", type=int, default=None,
              help="Stop after N consecutive empty polls (default: never).")
def worker(coord_url: str, experiment: str | None, imports: tuple[str, ...],
           backend: str, lease_seconds: int, max_idle: int | None) -> None:
    """Worker process — polls coordinator, runs shards."""
    from castra.worker import import_modules, run_forever
    if imports:
        import_modules(list(imports))
    n = run_forever(
        coord_url,
        backend=backend,
        experiment=experiment,
        lease_seconds=lease_seconds,
        max_idle_iterations=max_idle,
    )
    click.echo(f"worker exited; processed {n} shard(s).")


@main.group()
def workers() -> None:
    """Worker fleet operations (spawn / list / cost)."""


@workers.command(name="spawn-local")
@click.option("--coordinator", "coord_url", required=True,
              help="Coordinator URL, e.g. http://localhost:8765.")
@click.option("--count", type=int, default=None,
              help="Number of workers to spawn (default: cpu_count - 1).")
@click.option("--experiment", default=None,
              help="Restrict workers to shards from this experiment.")
@click.option("--import", "imports", multiple=True,
              help="Module to import in each worker. Repeatable.")
@click.option("--backend", default="local-subprocess", show_default=True,
              help="Backend label recorded with the worker registration.")
@click.option("--lease-seconds", default=240, show_default=True)
@click.option("--max-idle", type=int, default=None,
              help="Each worker stops after N consecutive empty polls.")
def workers_spawn_local(coord_url: str, count: int | None,
                        experiment: str | None, imports: tuple[str, ...],
                        backend: str, lease_seconds: int,
                        max_idle: int | None) -> None:
    """Spawn N local-subprocess workers and supervise them as a group."""
    from castra.fleet import auto_count, spawn_local_workers, supervise
    n = count if count is not None else auto_count()
    click.echo(f"spawning {n} local worker(s) -> {coord_url}")
    procs = spawn_local_workers(
        coord_url, n,
        experiment=experiment,
        imports=list(imports),
        backend=backend,
        lease_seconds=lease_seconds,
        max_idle=max_idle,
    )
    pids = ", ".join(str(p.pid) for p in procs)
    click.echo(f"  pids: {pids}")
    click.echo("  (Ctrl+C to stop)")
    code = supervise(procs)
    click.echo(f"all workers exited (max code {code})")


@workers.command(name="plan")
@click.option("--capacity", "capacity_path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Capacity YAML file.")
@click.option("--max-workers", type=int, required=True,
              help="Total worker target.")
def workers_plan(capacity_path: "Path", max_workers: int) -> None:
    """Print the fleet allocation for `max-workers` against the capacity config."""
    from castra.capacity import CapacityConfig, make_plan
    cap = CapacityConfig.from_yaml(capacity_path)
    plan = make_plan(cap, max_workers)
    click.echo(f"plan: {plan.total_workers} worker(s)")
    for u in plan.units:
        host = u.host or "(local)"
        click.echo(f"  {u.backend:<20} {host:<28} workers={u.workers}")
    if plan.total_workers < max_workers:
        click.echo(click.style(
            f"warning: capacity short by {max_workers - plan.total_workers} worker(s)",
            fg="yellow",
        ))


@workers.command(name="launch")
@click.option("--coordinator", "coord_url", required=True,
              help="Coordinator URL, e.g. http://localhost:8765.")
@click.option("--capacity", "capacity_path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Capacity YAML file.")
@click.option("--max-workers", type=int, required=True,
              help="Total worker target.")
@click.option("--experiment", default=None,
              help="Restrict workers to shards from this experiment.")
@click.option("--import", "imports", multiple=True,
              help="Module each worker should import. Repeatable.")
@click.option("--lease-seconds", default=240, show_default=True)
@click.option("--max-idle", type=int, default=None)
@click.option("--dry-run", is_flag=True,
              help="Print the plan without spawning anything.")
def workers_launch(coord_url: str, capacity_path: "Path", max_workers: int,
                   experiment: str | None, imports: tuple[str, ...],
                   lease_seconds: int, max_idle: int | None,
                   dry_run: bool) -> None:
    """Launch a mixed fleet (local + SSH) against the coordinator."""
    from castra.capacity import CapacityConfig, make_plan
    from castra.fleet import (
        spawn_local_workers, spawn_ssh_workers, supervise,
    )
    cap = CapacityConfig.from_yaml(capacity_path)
    plan = make_plan(cap, max_workers)

    click.echo(f"plan: {plan.total_workers} worker(s) -> {coord_url}")
    for u in plan.units:
        host = u.host or "(local)"
        click.echo(f"  {u.backend:<20} {host:<28} workers={u.workers}")
    if plan.total_workers < max_workers:
        click.echo(click.style(
            f"warning: capacity short by {max_workers - plan.total_workers} worker(s)",
            fg="yellow",
        ))
    if dry_run:
        return

    # Build host index for SSH lookup.
    host_by_name = {h.host: h for h in cap.ssh.hosts}

    procs = []
    next_idx = 0
    for u in plan.units:
        if u.backend == "local-subprocess":
            new = spawn_local_workers(
                coord_url, u.workers,
                experiment=experiment,
                imports=list(imports),
                lease_seconds=lease_seconds,
                max_idle=max_idle,
            )
        elif u.backend == "ssh":
            host_cfg = host_by_name[u.host]  # type: ignore[index]
            new = spawn_ssh_workers(
                coord_url, host_cfg, u.workers,
                experiment=experiment,
                imports=list(imports),
                lease_seconds=lease_seconds,
                max_idle=max_idle,
                starting_index=next_idx,
            )
        else:
            click.echo(click.style(
                f"  skipping unsupported backend: {u.backend}", fg="yellow"))
            continue
        next_idx += u.workers
        for p in new:
            click.echo(f"  spawned {u.backend} pid={p.pid} "
                       f"host={u.host or 'local'}")
        procs.extend(new)

    click.echo(f"  total spawned: {len(procs)} (Ctrl+C to stop)")
    code = supervise(procs)
    click.echo(f"all workers exited (max code {code})")


if __name__ == "__main__":
    main()
