"""castra command-line entry point."""

from __future__ import annotations

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


if __name__ == "__main__":
    main()
