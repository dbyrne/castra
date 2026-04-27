"""The worker process — single implementation, used across all backends.

A worker:
1. Registers with the coordinator and gets an ID.
2. Loops: claim a shard, deserialize via the protocol registry, run, report.
3. On failure, reports the exception and the shard is requeued.
4. Stops on Ctrl+C, exhaustion (configurable), or coordinator unreachable.

The same script runs locally, over SSH, and inside Docker on EC2 — only the
spawn method differs.
"""

from __future__ import annotations

import importlib
import socket
import time
import traceback
from typing import Callable

from castra import protocol
from castra.client import CoordinatorClient


def import_modules(modules: list[str]) -> None:
    """Import each module by name. User projects pass these in to populate
    the WorkUnit registry before claim loops start.
    """
    for mod in modules:
        importlib.import_module(mod)


def run_one(client: CoordinatorClient, worker_id: str, *,
            experiment: str | None = None,
            lease_seconds: int = 240) -> str | None:
    """Claim and run one shard. Returns the shard_id processed, or None if
    no work was available.
    """
    claim = client.claim(worker_id, experiment=experiment,
                         lease_seconds=lease_seconds)
    if claim is None:
        return None
    sid = claim["shard_id"]
    try:
        unit = protocol.deserialize(claim["type_name"], claim["payload"])
        result = unit.run()
    except Exception as e:  # noqa: BLE001 — worker must catch broad
        client.fail(
            sid,
            worker_id=worker_id,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            requeue=True,
        )
        return sid
    client.complete(sid, worker_id=worker_id, result=result)
    return sid


def run_forever(coordinator_url: str, *, backend: str = "local-subprocess",
                experiment: str | None = None,
                lease_seconds: int = 240,
                idle_sleep_s: float = 1.0,
                max_idle_iterations: int | None = None,
                stop_when: Callable[[], bool] | None = None) -> int:
    """Worker poll loop.

    `max_idle_iterations`: stop after this many consecutive empty claims
    (useful for `local-inproc` "drain to completion" style use). None = forever.

    `stop_when`: optional zero-arg callable returning True to stop the loop.

    Returns the number of shards processed.
    """
    processed = 0
    idle = 0
    with CoordinatorClient(coordinator_url) as client:
        worker_id = client.register_worker(
            backend=backend,
            hostname=socket.gethostname(),
        )
        try:
            while True:
                if stop_when is not None and stop_when():
                    break
                sid = run_one(
                    client, worker_id,
                    experiment=experiment,
                    lease_seconds=lease_seconds,
                )
                if sid is None:
                    idle += 1
                    if max_idle_iterations is not None and idle >= max_idle_iterations:
                        break
                    time.sleep(idle_sleep_s)
                    continue
                idle = 0
                processed += 1
                # Lightweight heartbeat after each completed shard.
                try:
                    client.heartbeat(worker_id)
                except Exception:  # noqa: BLE001
                    pass
        except KeyboardInterrupt:
            pass
    return processed
