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


if __name__ == "__main__":
    main()
