"""Local-subprocess worker fleet — spawn N `castra worker` processes
and supervise them as a group.

This is the simplest backend in the design's hierarchy:
- `local-inproc` (dev/test, no separate processes)
- `local-subprocess` ← *this module*
- `ssh` (Phase 5)
- `ec2-spot` / `ec2-ondemand` (Phase 7)

Each backend spawns the *same* worker process (`python -m castra.cli worker
...`); only how/where it gets started differs.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from castra.capacity import SSHHost


@dataclass
class WorkerProcess:
    index: int
    process: subprocess.Popen

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    @property
    def returncode(self) -> int | None:
        return self.process.returncode


def auto_count() -> int:
    """Default worker count: max(1, cpu_count - 1)."""
    return max(1, (os.cpu_count() or 1) - 1)


def worker_command(coordinator_url: str, *,
                   python: str | None = None,
                   experiment: str | None = None,
                   imports: list[str] | None = None,
                   backend: str = "local-subprocess",
                   lease_seconds: int = 240,
                   max_idle: int | None = None) -> list[str]:
    """Build the argv for spawning one castra worker subprocess.

    `python` defaults to the local interpreter; remote backends (SSH, EC2) pass
    in the remote machine's python path.

    Public so all backends share one builder.
    """
    cmd: list[str] = [
        python or sys.executable, "-m", "castra.cli", "worker",
        "--coordinator", coordinator_url,
        "--backend", backend,
        "--lease-seconds", str(lease_seconds),
    ]
    if experiment is not None:
        cmd += ["--experiment", experiment]
    if max_idle is not None:
        cmd += ["--max-idle", str(max_idle)]
    for mod in imports or []:
        cmd += ["--import", mod]
    return cmd


def ssh_command(host: SSHHost, remote_argv: list[str]) -> list[str]:
    """Wrap a remote command in `ssh user@host -- ...` with sensible defaults."""
    cmd: list[str] = ["ssh"]
    if host.ssh_key:
        cmd += ["-i", str(Path(host.ssh_key).expanduser())]
    cmd += [
        "-o", "BatchMode=yes",                 # no interactive prompts
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=60",
        f"{host.user}@{host.host}",
    ]
    cmd.append(shlex.join(remote_argv))
    return cmd


def spawn_local_workers(
    coordinator_url: str,
    count: int,
    *,
    experiment: str | None = None,
    imports: list[str] | None = None,
    backend: str = "local-subprocess",
    lease_seconds: int = 240,
    max_idle: int | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> list[WorkerProcess]:
    """Spawn `count` worker subprocesses on the local machine."""
    cmd = worker_command(
        coordinator_url,
        experiment=experiment,
        imports=imports,
        backend=backend,
        lease_seconds=lease_seconds,
        max_idle=max_idle,
    )
    workers: list[WorkerProcess] = []
    for i in range(count):
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env if env is not None else None,
        )
        workers.append(WorkerProcess(index=i, process=proc))
    return workers


def spawn_ssh_workers(
    coordinator_url: str,
    host: SSHHost,
    count: int,
    *,
    experiment: str | None = None,
    imports: list[str] | None = None,
    lease_seconds: int = 240,
    max_idle: int | None = None,
    starting_index: int = 0,
) -> list[WorkerProcess]:
    """Spawn `count` worker subprocesses on a remote SSH host.

    Each worker is its own ssh subprocess on the local machine; the supervise()
    loop treats them just like local subprocesses. The remote machine must have
    castra installed under `host.python_path`'s environment.
    """
    remote_cmd = worker_command(
        coordinator_url,
        python=host.python_path,
        experiment=experiment,
        imports=imports,
        backend=f"ssh:{host.host}",
        lease_seconds=lease_seconds,
        max_idle=max_idle,
    )
    local_cmd = ssh_command(host, remote_cmd)
    workers: list[WorkerProcess] = []
    for j in range(count):
        proc = subprocess.Popen(local_cmd)
        workers.append(WorkerProcess(index=starting_index + j, process=proc))
    return workers


def supervise(workers: list[WorkerProcess], *,
              poll_interval_s: float = 0.5,
              terminate_grace_s: float = 2.0) -> int:
    """Block until all workers exit. Returns the max exit code observed.

    On KeyboardInterrupt: terminate() all workers, wait up to terminate_grace_s,
    kill any stragglers.
    """
    try:
        while any(w.alive for w in workers):
            time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        for w in workers:
            try:
                w.process.terminate()
            except Exception:  # noqa: BLE001
                pass
        # Give them a moment to wind down cleanly.
        deadline = time.monotonic() + terminate_grace_s
        while any(w.alive for w in workers) and time.monotonic() < deadline:
            time.sleep(0.1)
        for w in workers:
            if w.alive:
                try:
                    w.process.kill()
                except Exception:  # noqa: BLE001
                    pass
    # Make sure poll() has set returncode.
    for w in workers:
        try:
            w.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
    codes = [w.returncode for w in workers if w.returncode is not None]
    return max(codes) if codes else 0
