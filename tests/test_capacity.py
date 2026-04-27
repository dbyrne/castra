"""Tests for CapacityConfig + make_plan."""

from __future__ import annotations

from pathlib import Path

import pytest

from castra.capacity import (
    CapacityConfig,
    FleetUnit,
    LocalCapacity,
    SSHCapacity,
    SSHHost,
    make_plan,
)


def test_capacity_from_dict_minimal() -> None:
    c = CapacityConfig.from_dict({})
    assert c.local.enabled is True
    assert c.local.workers == "auto"
    assert c.ssh.hosts == []


def test_capacity_from_dict_full() -> None:
    data = {
        "local": {"enabled": True, "workers": 4, "threads_per_worker": 2},
        "ssh": {
            "reachable_via": "tailscale",
            "hosts": [
                {"host": "h1", "user": "u1", "workers": 2,
                 "ssh_key": "~/.ssh/k", "python_path": "/opt/py"},
                {"host": "h2", "user": "u2", "workers": 8},
            ],
        },
    }
    c = CapacityConfig.from_dict(data)
    assert c.local.workers == 4
    assert c.local.threads_per_worker == 2
    assert c.ssh.reachable_via == "tailscale"
    assert len(c.ssh.hosts) == 2
    assert c.ssh.hosts[0].host == "h1"
    assert c.ssh.hosts[0].ssh_key == "~/.ssh/k"
    assert c.ssh.hosts[0].python_path == "/opt/py"
    assert c.ssh.hosts[1].workers == 8
    assert c.ssh.hosts[1].python_path == "python3"  # default


def test_capacity_unknown_top_level_raises() -> None:
    with pytest.raises(ValueError):
        CapacityConfig.from_dict({"oops": {}})


def test_local_resolved_workers_int() -> None:
    assert LocalCapacity(workers=8).resolved_workers() == 8


def test_local_resolved_workers_auto_at_least_one() -> None:
    assert LocalCapacity(workers="auto").resolved_workers() >= 1


def test_local_resolved_workers_invalid_raises() -> None:
    with pytest.raises(ValueError):
        LocalCapacity(workers="weird").resolved_workers()


def test_capacity_yaml_round_trip(tmp_path: Path) -> None:
    text = """
local:
  enabled: true
  workers: 4
ssh:
  reachable_via: lan
  hosts:
    - host: laptop-1
      user: david
      workers: 4
"""
    p = tmp_path / "capacity.yaml"
    p.write_text(text)
    c = CapacityConfig.from_yaml(p)
    assert c.local.workers == 4
    assert len(c.ssh.hosts) == 1
    assert c.ssh.hosts[0].host == "laptop-1"


# ---- plan generation ----


def test_plan_local_only_when_no_ssh() -> None:
    cap = CapacityConfig(local=LocalCapacity(workers=8))
    plan = make_plan(cap, max_workers=4)
    assert len(plan.units) == 1
    assert plan.units[0].backend == "local-subprocess"
    assert plan.units[0].workers == 4
    assert plan.total_workers == 4


def test_plan_caps_at_local_capacity() -> None:
    cap = CapacityConfig(local=LocalCapacity(workers=4))
    plan = make_plan(cap, max_workers=10)
    assert plan.total_workers == 4  # capped


def test_plan_local_then_ssh() -> None:
    cap = CapacityConfig(
        local=LocalCapacity(workers=4),
        ssh=SSHCapacity(hosts=[
            SSHHost(host="h1", user="u", workers=4),
            SSHHost(host="h2", user="u", workers=8),
        ]),
    )
    plan = make_plan(cap, max_workers=10)
    # 4 local + 4 from h1 + 2 from h2 = 10
    assert plan.total_workers == 10
    assert plan.units[0].backend == "local-subprocess"
    assert plan.units[0].workers == 4
    assert plan.units[1].backend == "ssh"
    assert plan.units[1].host == "h1"
    assert plan.units[1].workers == 4
    assert plan.units[2].backend == "ssh"
    assert plan.units[2].host == "h2"
    assert plan.units[2].workers == 2


def test_plan_skips_disabled_local() -> None:
    cap = CapacityConfig(
        local=LocalCapacity(enabled=False),
        ssh=SSHCapacity(hosts=[SSHHost(host="h1", user="u", workers=4)]),
    )
    plan = make_plan(cap, max_workers=4)
    assert all(u.backend != "local-subprocess" for u in plan.units)
    assert plan.total_workers == 4


def test_plan_short_when_capacity_insufficient() -> None:
    cap = CapacityConfig(local=LocalCapacity(workers=2))
    plan = make_plan(cap, max_workers=10)
    assert plan.total_workers == 2  # less than asked


def test_plan_zero_workers_returns_empty() -> None:
    cap = CapacityConfig(local=LocalCapacity(workers=4))
    plan = make_plan(cap, max_workers=0)
    assert plan.units == []
    assert plan.total_workers == 0


def test_fleet_plan_filter() -> None:
    plan_units = [
        FleetUnit("local-subprocess", host=None, workers=4),
        FleetUnit("ssh", host="h1", workers=2),
        FleetUnit("ssh", host="h2", workers=3),
    ]
    from castra.capacity import FleetPlan
    plan = FleetPlan(units=plan_units)
    ssh = plan.filter("ssh")
    assert len(ssh) == 2
    assert sum(u.workers for u in ssh) == 5
