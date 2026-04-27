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


@main.group()
def image() -> None:
    """Docker image build / push / run / stop / list."""


def _parse_kv(items: tuple[str, ...], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise click.ClickException(f"{flag} must be K=V, got {raw!r}")
        k, v = raw.split("=", 1)
        out[k] = v
    return out


@image.command(name="build")
@click.option("--tag", "-t", required=True,
              help="Image tag, e.g. my-agent:v1.")
@click.option("--context", "context_dir", default=".",
              show_default=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Build context directory.")
@click.option("--dockerfile", "-f", default=None,
              type=click.Path(path_type=Path),
              help="Path to Dockerfile (default: <context>/Dockerfile).")
@click.option("--build-arg", "build_args", multiple=True,
              help="Build-time variable, K=V. Repeatable.")
@click.option("--no-cache", is_flag=True, help="Skip layer cache.")
def image_build(tag: str, context_dir: "Path", dockerfile: "Path | None",
                build_args: tuple[str, ...], no_cache: bool) -> None:
    """Build a Docker image from a build context."""
    from castra.image import build as _build, DockerError
    try:
        rec = _build(
            context_dir, tag,
            dockerfile=dockerfile,
            build_args=_parse_kv(build_args, "--build-arg"),
            no_cache=no_cache,
        )
    except DockerError as e:
        raise click.ClickException(str(e))
    short = rec.image_id.split(":")[1][:12] if ":" in rec.image_id else rec.image_id[:12]
    click.echo(f"built {rec.tag} ({short})")


@image.command(name="push")
@click.argument("tag")
@click.option("--registry", default=None,
              help="Registry to push to (e.g. ghcr.io/user, docker.io/user).")
def image_push(tag: str, registry: str | None) -> None:
    """Push a local image to a registry."""
    from castra.image import push as _push, DockerError
    try:
        full = _push(tag, registry=registry)
    except DockerError as e:
        raise click.ClickException(str(e))
    click.echo(f"pushed {full}")


@image.command(name="run")
@click.argument("image_tag")
@click.option("--name", default=None, help="Container name.")
@click.option("--port", "-p", "ports", multiple=True,
              help="Port mapping HOST:CONTAINER. Repeatable.")
@click.option("--env", "-e", "envs", multiple=True,
              help="Environment variable K=V. Repeatable.")
@click.option("--detach/--no-detach", default=True, show_default=True)
@click.option("--rm", "auto_remove", is_flag=True,
              help="Auto-remove container on exit.")
def image_run(image_tag: str, name: str | None,
              ports: tuple[str, ...], envs: tuple[str, ...],
              detach: bool, auto_remove: bool) -> None:
    """Start a container from an image."""
    from castra.image import run as _run, DockerError
    parsed_ports: list[tuple[int, int]] = []
    for p in ports:
        if ":" not in p:
            raise click.ClickException(f"--port must be HOST:CONTAINER, got {p!r}")
        h, c = p.split(":", 1)
        try:
            parsed_ports.append((int(h), int(c)))
        except ValueError as e:
            raise click.ClickException(f"--port must be numeric: {p!r}") from e
    parsed_env = _parse_kv(envs, "--env")
    try:
        handle = _run(
            image_tag,
            ports=parsed_ports or None,
            env=parsed_env or None,
            name=name,
            detach=detach,
            auto_remove=auto_remove,
        )
    except DockerError as e:
        raise click.ClickException(str(e))
    click.echo(f"started {handle.name} ({handle.container_id[:12]})")
    if handle.ports:
        for c_port, h_port in sorted(handle.ports.items()):
            click.echo(f"  port: localhost:{h_port} -> container:{c_port}")


@image.command(name="stop")
@click.argument("name_or_id")
@click.option("--timeout", default=10, show_default=True,
              help="Seconds to wait for clean shutdown before SIGKILL.")
@click.option("--keep", is_flag=True,
              help="Stop but don't remove the container.")
def image_stop(name_or_id: str, timeout: int, keep: bool) -> None:
    """Stop (and remove) a container."""
    from castra.image import stop as _stop, DockerError
    try:
        _stop(name_or_id, timeout=timeout, remove=not keep)
    except DockerError as e:
        raise click.ClickException(str(e))
    click.echo(f"stopped {name_or_id}")


@image.command(name="list")
@click.option("--all", "-a", "include_stopped", is_flag=True,
              help="Include stopped containers.")
def image_list(include_stopped: bool) -> None:
    """List containers (running by default)."""
    from castra.image import list_containers, DockerError
    try:
        rows = list_containers(include_stopped=include_stopped)
    except DockerError as e:
        raise click.ClickException(str(e))
    if not rows:
        click.echo("no containers")
        return
    click.echo(f"  {'NAME':<22}  {'IMAGE':<32}  STATUS")
    click.echo(f"  {'-'*22}  {'-'*32}  ------")
    for c in rows:
        click.echo(f"  {c.name:<22}  {c.image:<32}  {c.status}")


if __name__ == "__main__":
    main()
