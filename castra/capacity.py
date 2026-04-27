"""Capacity configuration: declares the compute available to spawn workers on.

A single YAML describes local / SSH / (later: EC2) capacity. Fleet planning
reads this + a target worker count and decides who launches what. Local is
filled first (zero cost), then SSH hosts in declared order, then EC2.

Schema:

    local:
      enabled: true
      workers: auto              # int or "auto" (= cpu_count - 1)
      threads_per_worker: 1

    ssh:
      reachable_via: tailscale   # lan | tailscale | (informational only)
      hosts:
        - host: laptop-1
          user: david
          ssh_key: ~/.ssh/id_ed25519
          workers: 4
          python_path: ~/projectvenv/bin/python
        - host: laptop-2
          user: david
          workers: 8
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LocalCapacity:
    enabled: bool = True
    workers: int | str = "auto"  # int or "auto"
    threads_per_worker: int = 1

    def resolved_workers(self) -> int:
        if isinstance(self.workers, int):
            return max(0, self.workers)
        if self.workers == "auto":
            return max(1, (os.cpu_count() or 1) - 1)
        raise ValueError(f"unknown 'workers' value: {self.workers!r}")


@dataclass
class SSHHost:
    host: str
    user: str
    ssh_key: str | None = None
    workers: int = 1
    threads_per_worker: int = 1
    python_path: str = "python3"


@dataclass
class SSHCapacity:
    reachable_via: str = "lan"          # lan | tailscale | other (informational)
    hosts: list[SSHHost] = field(default_factory=list)


@dataclass
class CapacityConfig:
    local: LocalCapacity = field(default_factory=LocalCapacity)
    ssh: SSHCapacity = field(default_factory=SSHCapacity)
    # ec2: EC2Capacity will land in Phase 7.

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapacityConfig":
        if data is None:
            return cls()
        local_data = data.get("local") or {}
        ssh_data = data.get("ssh") or {}
        local = LocalCapacity(
            enabled=local_data.get("enabled", True),
            workers=local_data.get("workers", "auto"),
            threads_per_worker=local_data.get("threads_per_worker", 1),
        )
        hosts = []
        for h in ssh_data.get("hosts") or []:
            hosts.append(SSHHost(
                host=h["host"],
                user=h.get("user") or os.environ.get("USER") or "root",
                ssh_key=h.get("ssh_key"),
                workers=int(h.get("workers", 1)),
                threads_per_worker=int(h.get("threads_per_worker", 1)),
                python_path=h.get("python_path", "python3"),
            ))
        ssh = SSHCapacity(
            reachable_via=ssh_data.get("reachable_via", "lan"),
            hosts=hosts,
        )
        unknown = set(data) - {"local", "ssh"}
        if unknown:
            raise ValueError(f"unknown top-level capacity keys: {sorted(unknown)}")
        return cls(local=local, ssh=ssh)

    @classmethod
    def from_yaml(cls, path: Path) -> "CapacityConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})


@dataclass
class FleetUnit:
    """One contiguous group of workers on a single backend/host."""
    backend: str           # "local-subprocess" | "ssh" | (later) "ec2-spot" | ...
    host: str | None       # None for local
    workers: int           # number of worker processes in this unit
    hourly_cost: float = 0.0  # for cost rollups; local + SSH are 0


@dataclass
class FleetPlan:
    units: list[FleetUnit] = field(default_factory=list)

    @property
    def total_workers(self) -> int:
        return sum(u.workers for u in self.units)

    @property
    def total_hourly_cost(self) -> float:
        return sum(u.hourly_cost for u in self.units)

    def filter(self, backend: str) -> list[FleetUnit]:
        return [u for u in self.units if u.backend == backend]


def make_plan(capacity: CapacityConfig, max_workers: int) -> FleetPlan:
    """Allocate up to `max_workers` across declared capacity, cheapest first.

    Order: local → SSH (in declared order) → EC2 (Phase 7).
    Local workers are bounded by `capacity.local.resolved_workers()` (typically
    cpu_count - 1). SSH hosts are bounded by their per-host `workers`.
    """
    units: list[FleetUnit] = []
    remaining = max_workers

    if capacity.local.enabled and remaining > 0:
        local_cap = capacity.local.resolved_workers()
        n = min(local_cap, remaining)
        if n > 0:
            units.append(FleetUnit("local-subprocess", host=None, workers=n))
            remaining -= n

    for h in capacity.ssh.hosts:
        if remaining <= 0:
            break
        n = min(h.workers, remaining)
        if n > 0:
            units.append(FleetUnit("ssh", host=h.host, workers=n))
            remaining -= n

    return FleetPlan(units=units)
