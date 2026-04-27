"""Tests for local-subprocess worker fleet supervision."""

from __future__ import annotations

import itertools
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from castra.capacity import SSHHost
from castra.fleet import (
    WorkerProcess,
    auto_count,
    spawn_local_workers,
    spawn_ssh_workers,
    ssh_command,
    supervise,
    worker_command,
)


def _mock_proc(exit_code: int, alive_iters: int = 2) -> MagicMock:
    """Mock subprocess that's 'alive' for `alive_iters` polls then exits with code."""
    m = MagicMock()
    m.poll = MagicMock(side_effect=itertools.chain(
        [None] * alive_iters, itertools.repeat(exit_code)
    ))
    m.returncode = exit_code
    m.wait = MagicMock(return_value=exit_code)
    m.terminate = MagicMock()
    m.kill = MagicMock()
    return m


def test_auto_count_at_least_one() -> None:
    assert auto_count() >= 1


def test_worker_command_includes_required_flags() -> None:
    cmd = worker_command(
        "http://localhost:8765",
        experiment="exp-1",
        imports=["foo.bar"],
        backend="test",
        lease_seconds=120,
        max_idle=5,
    )
    assert sys.executable in cmd
    assert "-m" in cmd
    assert "castra.cli" in cmd
    assert "worker" in cmd
    assert "--coordinator" in cmd
    i = cmd.index("--coordinator")
    assert cmd[i + 1] == "http://localhost:8765"
    assert "--experiment" in cmd
    assert "exp-1" in cmd
    assert "--import" in cmd
    assert "foo.bar" in cmd
    assert "--backend" in cmd
    assert "test" in cmd
    assert "--max-idle" in cmd
    assert "5" in cmd


def test_worker_command_omits_optional_when_none() -> None:
    cmd = worker_command("http://x", experiment=None, imports=[],
                         max_idle=None)
    assert "--experiment" not in cmd
    assert "--import" not in cmd
    assert "--max-idle" not in cmd


def test_spawn_local_workers_creates_processes() -> None:
    """Mock subprocess.Popen and verify spawn_local_workers creates count procs."""
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.poll = MagicMock(return_value=0)
    with patch("castra.fleet.subprocess.Popen", return_value=mock_proc) as popen:
        workers = spawn_local_workers("http://x", count=3, imports=["a.b"])
        assert len(workers) == 3
        assert popen.call_count == 3
        # All should share the same command shape
        first_call_args = popen.call_args_list[0][0][0]
        assert "--coordinator" in first_call_args
        assert "http://x" in first_call_args
        assert "--import" in first_call_args
        assert "a.b" in first_call_args


def test_supervise_returns_max_exit_code() -> None:
    procs = [
        WorkerProcess(index=0, process=_mock_proc(0)),
        WorkerProcess(index=1, process=_mock_proc(1)),
        WorkerProcess(index=2, process=_mock_proc(0)),
    ]
    rc = supervise(procs, poll_interval_s=0.01)
    assert rc == 1


def test_supervise_returns_zero_when_all_succeed() -> None:
    procs = [WorkerProcess(index=i, process=_mock_proc(0)) for i in range(3)]
    rc = supervise(procs, poll_interval_s=0.01)
    assert rc == 0


def test_supervise_handles_keyboard_interrupt() -> None:
    """On KeyboardInterrupt, all workers should be terminated."""
    procs = [
        WorkerProcess(index=i, process=_mock_proc(0, alive_iters=10))
        for i in range(2)
    ]

    sleep_calls = {"n": 0}
    real_sleep = time.sleep

    def fake_sleep(s):
        sleep_calls["n"] += 1
        if sleep_calls["n"] == 1:
            raise KeyboardInterrupt
        real_sleep(0.001)

    with patch("castra.fleet.time.sleep", side_effect=fake_sleep):
        supervise(procs, terminate_grace_s=0.05)

    for w in procs:
        w.process.terminate.assert_called_once()


def test_worker_process_pid_and_alive_properties() -> None:
    m = MagicMock()
    m.pid = 4242
    m.poll = MagicMock(return_value=None)
    m.returncode = None
    w = WorkerProcess(index=0, process=m)
    assert w.pid == 4242
    assert w.alive is True
    m.poll = MagicMock(return_value=0)
    m.returncode = 0
    assert w.alive is False


# ---- SSH backend ----


def test_worker_command_uses_provided_python() -> None:
    cmd = worker_command("http://x", python="/opt/python/bin/python")
    assert cmd[0] == "/opt/python/bin/python"


def test_ssh_command_includes_host_user() -> None:
    host = SSHHost(host="myhost", user="alice", ssh_key="~/.ssh/id_rsa")
    cmd = ssh_command(host, ["python3", "-m", "castra.cli", "worker",
                              "--coordinator", "http://x"])
    assert cmd[0] == "ssh"
    assert "alice@myhost" in cmd
    # The remote command is shlex-joined as the last argument.
    assert "castra.cli" in cmd[-1]


def test_ssh_command_includes_ssh_key_when_provided() -> None:
    host = SSHHost(host="h", user="u", ssh_key="/keys/id")
    cmd = ssh_command(host, ["echo", "hi"])
    assert "-i" in cmd
    i = cmd.index("-i")
    # The path may be expanded but the leaf 'id' should remain.
    assert "id" in cmd[i + 1]


def test_ssh_command_omits_key_when_none() -> None:
    host = SSHHost(host="h", user="u", ssh_key=None)
    cmd = ssh_command(host, ["echo", "hi"])
    assert "-i" not in cmd


def test_ssh_command_uses_batch_mode() -> None:
    """BatchMode=yes prevents interactive prompts (passwords, host-key check)."""
    host = SSHHost(host="h", user="u")
    cmd = ssh_command(host, ["echo", "hi"])
    # Concatenated form: should appear verbatim somewhere as adjacent items.
    options = " ".join(cmd)
    assert "BatchMode=yes" in options


def test_ssh_command_quotes_remote_argv() -> None:
    """A remote arg containing spaces should be properly quoted."""
    host = SSHHost(host="h", user="u")
    cmd = ssh_command(host, ["python", "-c", "print('hello world')"])
    # Last element is the joined remote command; should retain quoting.
    assert "'hello world'" in cmd[-1] or '"hello world"' in cmd[-1] or "'\\''" in cmd[-1]


def test_spawn_ssh_workers_uses_remote_python() -> None:
    """spawn_ssh_workers should embed the host's python_path in the remote cmd."""
    host = SSHHost(host="h", user="u", workers=2,
                   python_path="/opt/proj/.venv/bin/python")
    with patch("castra.fleet.subprocess.Popen") as popen:
        popen.return_value = MagicMock(pid=999, poll=lambda: None)
        spawn_ssh_workers("http://x", host, count=2,
                          imports=["foedus.shards"])
        assert popen.call_count == 2
        argv = popen.call_args_list[0][0][0]
        assert argv[0] == "ssh"
        # The shlex-joined remote command (last element) must include remote python.
        assert "/opt/proj/.venv/bin/python" in argv[-1]
        assert "foedus.shards" in argv[-1]


def test_spawn_ssh_workers_records_backend_label() -> None:
    """The worker registers as backend='ssh:<host>' so cost rollups can attribute."""
    host = SSHHost(host="laptop-basement", user="u", workers=1)
    with patch("castra.fleet.subprocess.Popen") as popen:
        popen.return_value = MagicMock(pid=1, poll=lambda: None)
        spawn_ssh_workers("http://x", host, count=1)
        argv = popen.call_args_list[0][0][0]
        # remote command (last element) contains --backend ssh:laptop-basement
        assert "ssh:laptop-basement" in argv[-1]
