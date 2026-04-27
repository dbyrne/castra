"""Live TUI dashboard for castra.

Two stacked panes (kept simple intentionally for v1):

  +-----------------------------------------------+
  | Experiments                                   |
  |  name        status     branch         path    |
  |  exp-1       active     experiment/.   ...     |
  +-----------------------------------------------+
  | Coordinator                                    |
  |  url:    http://localhost:8765                |
  |  shards: pending=8 claimed=4 completed=42     |
  |  workers: 4 active                            |
  +-----------------------------------------------+

Polls the coordinator at 1 Hz when a URL is provided. Without one, just
shows the experiment list.

The data layer (`fetch_snapshot`) is separate from rendering so it's
easy to test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from castra import paths, worktree
from castra.client import CoordinatorClient
from castra.spec import ExperimentSpec


@dataclass
class DashboardSnapshot:
    experiments: list[dict[str, Any]] = field(default_factory=list)
    coordinator_url: str | None = None
    healthz: dict[str, Any] | None = None
    workers: list[dict[str, Any]] = field(default_factory=list)
    coordinator_error: str | None = None


def _experiment_rows(project_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for w in worktree.list_all(project_root=project_root):
        config_path = w.path / "experiments" / w.name / "config.yaml"
        status = "active"
        parent: str | None = None
        axes: list[str] = []
        if config_path.exists():
            try:
                spec = ExperimentSpec.from_yaml(config_path)
                if spec.concluded_gen is not None:
                    status = "final"
                parent = spec.parent
                axes = sorted(spec.axes.keys())
            except Exception:  # noqa: BLE001
                status = "unreadable"
        elif not w.path.exists():
            status = "archived"
        out.append({
            "name": w.name,
            "status": status,
            "branch": w.branch,
            "path": str(w.path),
            "parent": parent,
            "axes": axes,
        })
    # Sort: active first, then final, then archived/unreadable.
    order = {"active": 0, "final": 1, "archived": 2, "unreadable": 3}
    out.sort(key=lambda r: (order.get(r["status"], 99), r["name"]))
    return out


def fetch_snapshot(coordinator_url: str | None = None,
                   project_root: Path | None = None) -> DashboardSnapshot:
    """Build a fresh snapshot. Survives coordinator unreachability."""
    snap = DashboardSnapshot(coordinator_url=coordinator_url)
    try:
        snap.experiments = _experiment_rows(project_root or paths.project_root())
    except Exception as e:  # noqa: BLE001
        # Likely "not in a git repo" — surface but don't crash.
        snap.experiments = []
        snap.coordinator_error = f"experiment list failed: {e}"
    if coordinator_url:
        try:
            with CoordinatorClient(coordinator_url, timeout=2.0) as c:
                snap.healthz = c.healthz()
                # Pull worker list via experiment_label if available.
                label = (snap.healthz or {}).get("experiment_label")
                if label:
                    status = c.status(label)
                    snap.workers = status.get("workers", [])
        except (httpx.RequestError, httpx.HTTPStatusError, OSError) as e:
            snap.coordinator_error = f"coordinator unreachable: {e}"
    return snap


def _render_experiments(snap: DashboardSnapshot) -> Panel:
    if not snap.experiments:
        body = Text("no experiments yet", style="dim")
        return Panel(body, title="experiments", border_style="cyan")
    table = Table(box=None, expand=True, padding=(0, 1))
    table.add_column("name", style="bold")
    table.add_column("status")
    table.add_column("axes", style="cyan")
    table.add_column("parent")
    table.add_column("branch", style="dim")
    for e in snap.experiments:
        status_style = {
            "active": "green",
            "final": "yellow",
            "archived": "dim",
            "unreadable": "red",
        }.get(e["status"], "")
        table.add_row(
            e["name"],
            Text(e["status"], style=status_style),
            ",".join(e["axes"]) or "-",
            e["parent"] or "-",
            e["branch"],
        )
    return Panel(table, title=f"experiments ({len(snap.experiments)})",
                 border_style="cyan")


def _render_coordinator(snap: DashboardSnapshot) -> Panel:
    if snap.coordinator_url is None:
        body = Text("(no --coordinator URL given; use --coordinator http://...)",
                    style="dim")
        return Panel(body, title="coordinator", border_style="magenta")
    if snap.coordinator_error:
        return Panel(Text(snap.coordinator_error, style="red"),
                     title="coordinator", border_style="red")
    if snap.healthz is None:
        return Panel(Text("(connecting...)", style="dim"),
                     title="coordinator", border_style="magenta")
    counts = snap.healthz.get("shard_counts", {})
    label = snap.healthz.get("experiment_label") or "(unset)"
    parts = [
        f"url:        {snap.coordinator_url}",
        f"experiment: {label}",
        f"shards:     "
        + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        or "shards: (none)",
    ]
    body_lines = [Text(line) for line in parts]
    if snap.workers:
        body_lines.append(Text(""))
        body_lines.append(Text(f"workers ({len(snap.workers)}):", style="bold"))
        for w in snap.workers[:20]:  # cap display
            body_lines.append(Text(
                f"  {w.get('worker_id', '?')[:8]}  "
                f"{w.get('backend', '?'):<18}  "
                f"{w.get('hostname', '?'):<24}  "
                f"hb {w.get('last_heartbeat_at', '')}"
            ))
    else:
        body_lines.append(Text(""))
        body_lines.append(Text("workers: (none registered)", style="dim"))
    return Panel(Group(*body_lines), title="coordinator",
                 border_style="magenta")


def render(snap: DashboardSnapshot) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="experiments", ratio=2),
        Layout(name="coordinator", ratio=1),
    )
    layout["experiments"].update(_render_experiments(snap))
    layout["coordinator"].update(_render_coordinator(snap))
    return layout


def run(coordinator_url: str | None, *, refresh_hz: float = 1.0,
        project_root: Path | None = None) -> None:
    """Run the dashboard until Ctrl+C."""
    console = Console()
    period = 1.0 / max(refresh_hz, 0.1)
    try:
        with Live(render(fetch_snapshot(coordinator_url, project_root)),
                  console=console,
                  refresh_per_second=refresh_hz,
                  screen=True) as live:
            while True:
                time.sleep(period)
                snap = fetch_snapshot(coordinator_url, project_root)
                live.update(render(snap))
    except KeyboardInterrupt:
        pass
