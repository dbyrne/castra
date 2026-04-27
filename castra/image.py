"""Generic Docker image primitives — build, push, run, stop, list.

Shells out to the `docker` CLI rather than using the Python SDK. Three reasons:
- The Python `docker` package has its own dependency tree (paramiko, urllib3
  pinning, etc.) that's brittle.
- Most environments where you'd run castra already have the docker CLI.
- Substituting `podman` later is a matter of swapping a single command name.

Registry-agnostic: dockerhub, GHCR, ECR, local registry — same API.
ECR-specific multi-region push and IAM glue lands later, when EC2 training
returns the cost question.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DOCKER_BIN = "docker"


class DockerError(RuntimeError):
    """A docker CLI invocation failed (or docker isn't on PATH)."""


@dataclass
class ImageRecord:
    tag: str
    image_id: str   # full sha256:... id


@dataclass
class ContainerHandle:
    container_id: str
    name: str
    image: str
    ports: dict[int, int] = field(default_factory=dict)  # container_port -> host_port


@dataclass
class ContainerInfo:
    container_id: str
    name: str
    image: str
    status: str


def _run(argv: list[str], *, check: bool = True) -> str:
    """Run a docker subcommand and return stdout (stripped of trailing newline).

    Raises DockerError on non-zero exit or if docker isn't installed.
    """
    if shutil.which(argv[0]) is None:
        raise DockerError(f"{argv[0]!r} not found on PATH; install Docker first")
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=check,
        )
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        raise DockerError(
            f"{' '.join(argv)} exited {e.returncode}: {msg}"
        ) from e
    return result.stdout


def build(
    context_dir: Path | str,
    tag: str,
    *,
    dockerfile: Path | str | None = None,
    build_args: dict[str, str] | None = None,
    no_cache: bool = False,
    quiet: bool = False,
) -> ImageRecord:
    """Build an image. Returns an ImageRecord with the built image's full ID."""
    cmd = [DOCKER_BIN, "build", "-t", tag]
    if dockerfile is not None:
        cmd.extend(["-f", str(dockerfile)])
    if build_args:
        for k, v in build_args.items():
            cmd.extend(["--build-arg", f"{k}={v}"])
    if no_cache:
        cmd.append("--no-cache")
    if quiet:
        cmd.append("--quiet")
    cmd.append(str(context_dir))
    _run(cmd)

    image_id = _run([
        DOCKER_BIN, "image", "inspect", "--format", "{{.Id}}", tag,
    ]).strip()
    return ImageRecord(tag=tag, image_id=image_id)


def push(tag: str, registry: str | None = None) -> str:
    """Tag (if `registry` provided) and push the image. Returns the pushed URI."""
    if registry:
        full_tag = f"{registry.rstrip('/')}/{tag}"
        _run([DOCKER_BIN, "tag", tag, full_tag])
    else:
        full_tag = tag
    _run([DOCKER_BIN, "push", full_tag])
    return full_tag


def run(
    image: str,
    *,
    ports: list[tuple[int, int]] | None = None,   # [(host, container), ...]
    env: dict[str, str] | None = None,
    name: str | None = None,
    detach: bool = True,
    auto_remove: bool = False,
    extra_args: list[str] | None = None,
    command: list[str] | None = None,
) -> ContainerHandle:
    """Start a container. Returns a ContainerHandle."""
    cmd = [DOCKER_BIN, "run"]
    if detach:
        cmd.append("-d")
    if auto_remove:
        cmd.append("--rm")
    if name:
        cmd.extend(["--name", name])
    if ports:
        for host_port, container_port in ports:
            cmd.extend(["-p", f"{host_port}:{container_port}"])
    if env:
        for k, v in env.items():
            cmd.extend(["-e", f"{k}={v}"])
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(image)
    if command:
        cmd.extend(command)

    container_id = _run(cmd).strip()
    return ContainerHandle(
        container_id=container_id,
        name=name or container_id[:12],
        image=image,
        ports={c: h for h, c in (ports or [])},
    )


def stop(name_or_id: str, *, timeout: int = 10, remove: bool = True) -> None:
    """Stop a container (and `docker rm` it by default)."""
    _run([DOCKER_BIN, "stop", "-t", str(timeout), name_or_id])
    if remove:
        # If --rm was used at run time the container may already be gone;
        # tolerate that case rather than failing the stop.
        try:
            _run([DOCKER_BIN, "rm", name_or_id])
        except DockerError:
            pass


def list_containers(*, include_stopped: bool = False) -> list[ContainerInfo]:
    """List containers. By default only running ones."""
    cmd = [DOCKER_BIN, "ps", "--format", "{{json .}}"]
    if include_stopped:
        cmd.append("-a")
    out = _run(cmd)
    rows: list[ContainerInfo] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        rows.append(ContainerInfo(
            container_id=data.get("ID", ""),
            name=data.get("Names", ""),
            image=data.get("Image", ""),
            status=data.get("Status", ""),
        ))
    return rows


def list_images() -> list[dict]:
    """List local images. Returns docker's raw JSON dicts (untyped on purpose
    since the schema is wide and we don't need to introspect every field)."""
    out = _run([DOCKER_BIN, "image", "ls", "--format", "{{json .}}"])
    rows = []
    for line in out.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def inspect_image(tag: str) -> dict:
    """Return docker inspect output for a tag, parsed as a dict."""
    out = _run([DOCKER_BIN, "image", "inspect", tag])
    parsed = json.loads(out)
    if not parsed:
        raise DockerError(f"no such image: {tag!r}")
    return parsed[0]
