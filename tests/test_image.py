"""Tests for the docker primitives in castra/image.py.

We mock the underlying `_run` helper so tests don't shell out to docker —
keeps them fast and CI-friendly. The tests verify command construction
(the part most likely to drift) and parsing (the part that translates
docker output into our dataclasses).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from castra import image as image_mod
from castra.image import (
    ContainerInfo,
    DockerError,
    build,
    inspect_image,
    list_containers,
    push,
    run,
    stop,
)


@pytest.fixture
def fake_run():
    """Patch castra.image._run with a Mock that records calls.

    Default return is "". Tests can override:
      - fake_run.return_value = "..."     (every call returns the same)
      - fake_run.side_effect = [...]      (queue per-call returns)
    Use fake_run.call_args_list to inspect what was sent to docker.
    """
    with patch("castra.image._run") as m:
        m.return_value = ""
        yield m


def test_build_basic_command(fake_run) -> None:
    fake_run.side_effect = ["", "sha256:abc12345\n"]
    rec = build("/ctx", "myimg:v1")
    assert rec.tag == "myimg:v1"
    assert rec.image_id == "sha256:abc12345"

    build_cmd = fake_run.call_args_list[0].args[0]
    assert build_cmd[0] == "docker"
    assert build_cmd[1] == "build"
    assert "-t" in build_cmd
    assert "myimg:v1" in build_cmd
    assert "/ctx" in build_cmd

    inspect_cmd = fake_run.call_args_list[1].args[0]
    assert inspect_cmd[:3] == ["docker", "image", "inspect"]


def test_build_with_dockerfile_and_args(fake_run) -> None:
    fake_run.side_effect = ["", "sha256:x"]
    build(
        "/ctx", "img:1",
        dockerfile=Path("/path/Dockerfile.agent"),
        build_args={"AGENT": "foo.bar", "VERSION": "v3"},
        no_cache=True,
    )
    build_cmd = fake_run.call_args_list[0].args[0]
    assert "-f" in build_cmd
    fi = build_cmd.index("-f")
    assert build_cmd[fi + 1].endswith("Dockerfile.agent")
    # Build args appear as --build-arg K=V pairs (order may vary).
    pairs = [
        build_cmd[i + 1] for i, t in enumerate(build_cmd) if t == "--build-arg"
    ]
    assert "AGENT=foo.bar" in pairs
    assert "VERSION=v3" in pairs
    assert "--no-cache" in build_cmd


def test_push_with_registry_tags_first(fake_run) -> None:
    fake_run.return_value = ""
    full = push("img:1", registry="ghcr.io/me")
    assert full == "ghcr.io/me/img:1"
    cmds = [c.args[0] for c in fake_run.call_args_list]
    assert cmds[0][:2] == ["docker", "tag"]
    assert cmds[0][-2:] == ["img:1", "ghcr.io/me/img:1"]
    assert cmds[1] == ["docker", "push", "ghcr.io/me/img:1"]


def test_push_no_registry_uses_tag_directly(fake_run) -> None:
    fake_run.return_value = ""
    full = push("img:1")
    assert full == "img:1"
    cmds = [c.args[0] for c in fake_run.call_args_list]
    # No tag command, single push.
    assert cmds == [["docker", "push", "img:1"]]


def test_run_basic_returns_handle(fake_run) -> None:
    fake_run.return_value = "abc123def456\n"
    handle = run("img:1", name="myc")
    assert handle.container_id == "abc123def456"
    assert handle.name == "myc"
    assert handle.image == "img:1"
    cmd = fake_run.call_args_list[0].args[0]
    assert cmd[:3] == ["docker", "run", "-d"]
    assert "--name" in cmd
    assert "myc" in cmd
    assert "img:1" in cmd


def test_run_with_ports_and_env(fake_run) -> None:
    fake_run.return_value = "container-id\n"
    run(
        "img:1",
        ports=[(8080, 80), (9000, 9000)],
        env={"FOO": "bar", "X": "y"},
        auto_remove=True,
    )
    cmd = fake_run.call_args_list[0].args[0]
    # Port mappings
    port_pairs = [cmd[i + 1] for i, t in enumerate(cmd) if t == "-p"]
    assert "8080:80" in port_pairs
    assert "9000:9000" in port_pairs
    # Env vars
    env_pairs = [cmd[i + 1] for i, t in enumerate(cmd) if t == "-e"]
    assert "FOO=bar" in env_pairs
    assert "X=y" in env_pairs
    assert "--rm" in cmd


def test_run_no_detach(fake_run) -> None:
    fake_run.return_value = ""
    run("img:1", detach=False)
    cmd = fake_run.call_args_list[0].args[0]
    assert "-d" not in cmd


def test_run_handle_records_port_mapping(fake_run) -> None:
    fake_run.return_value = "id\n"
    handle = run("img:1", ports=[(8080, 80)])
    assert handle.ports == {80: 8080}


def test_stop_calls_stop_then_rm(fake_run) -> None:
    fake_run.return_value = ""
    stop("myc")
    cmds = [c.args[0] for c in fake_run.call_args_list]
    assert cmds[0] == ["docker", "stop", "-t", "10", "myc"]
    assert cmds[1] == ["docker", "rm", "myc"]


def test_stop_keep_skips_rm(fake_run) -> None:
    fake_run.return_value = ""
    stop("myc", remove=False)
    cmds = [c.args[0] for c in fake_run.call_args_list]
    assert len(cmds) == 1
    assert cmds[0] == ["docker", "stop", "-t", "10", "myc"]


def test_stop_tolerates_missing_container_on_rm(fake_run) -> None:
    """If `docker rm` fails (container already gone with --rm), don't crash."""
    fake_run.side_effect = ["", DockerError("no such container")]
    stop("myc")  # should not raise


def test_list_containers_parses_json_lines(fake_run) -> None:
    fake_run.return_value = (
        '{"ID":"a1","Names":"foo","Image":"img:1","Status":"Up 2 minutes"}\n'
        '{"ID":"b2","Names":"bar","Image":"img:2","Status":"Exited (0) 1 minute ago"}\n'
    )
    rows = list_containers(include_stopped=True)
    assert len(rows) == 2
    assert rows[0].container_id == "a1"
    assert rows[0].name == "foo"
    assert rows[1].image == "img:2"


def test_list_containers_empty(fake_run) -> None:
    fake_run.return_value = ""
    assert list_containers() == []


def test_inspect_image_returns_first_object(fake_run) -> None:
    fake_run.return_value = '[{"Id":"sha256:x","Architecture":"amd64"}]'
    info = inspect_image("img:1")
    assert info["Id"] == "sha256:x"


def test_inspect_image_missing_raises(fake_run) -> None:
    fake_run.return_value = "[]"
    with pytest.raises(DockerError):
        inspect_image("ghost")


# ---- Error path: docker missing or non-zero exit ----


def test_run_helper_raises_when_docker_missing(monkeypatch) -> None:
    monkeypatch.setattr("castra.image.shutil.which", lambda _: None)
    with pytest.raises(DockerError) as excinfo:
        image_mod._run(["docker", "ps"])
    assert "not found on PATH" in str(excinfo.value)


def test_run_helper_raises_on_nonzero_exit(monkeypatch) -> None:
    monkeypatch.setattr("castra.image.shutil.which", lambda _: "/usr/bin/docker")

    def fake_subprocess(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=125, cmd=args[0], output="", stderr="bad image\n"
        )

    monkeypatch.setattr("castra.image.subprocess.run", fake_subprocess)
    with pytest.raises(DockerError) as excinfo:
        image_mod._run(["docker", "build", "/x"])
    assert "exited 125" in str(excinfo.value)
    assert "bad image" in str(excinfo.value)
