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
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


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
                   experiment: str | None = None,
                   imports: list[str] | None = None,
                   backend: str = "local-subprocess",
                   lease_seconds: int = 240,
                   max_idle: int | None = None) -> list[str]:
    """Build the argv for spawning one castra worker subprocess.

    Public so tests + remote backends (SSH, EC2) can reuse the same builder.
    """
    cmd: list[str] = [
        sys.executable, "-m", "castra.cli", "worker",
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
    """Spawn `count` worker subprocesses. Returns the list."""
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
