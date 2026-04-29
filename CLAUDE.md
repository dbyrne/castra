# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` is the elevator pitch and `DESIGN.md` is the architectural intent (status: design phase). This file covers what's specifically load-bearing when changing code.

**Note**: `DESIGN.md` says "design phase, no code yet." That's stale — the package now has ~17 modules and a populated test suite. Treat `DESIGN.md` as **architectural intent**, not a current-state snapshot. The build-phase ordering at the bottom (1: worktree lifecycle, 2: coordinator + local-inproc, …, 8: compare/promote polish) still describes the priority order.

## Commands

```sh
pip install -e .[dev]
pytest                                 # all tests; one test file per module under castra/
pytest tests/test_worktree.py          # single module
pytest tests/test_coordinator.py -k claim   # filter
```

CLI entry point is `castra = castra.cli:main`. Click groups:

- `castra exp create|fork|repair|finalize|ship|promote|archive|list|info|compare|dashboard` — experiment lifecycle
- `castra coordinator start|status|stop` — fastapi sidecar that owns shard state
- `castra workers plan|launch|cost|list` — fleet management across backends
- `castra worker-image build|list` — Docker images for the EC2 backend
- `castra worker --coordinator <url>` — the actual worker process (one impl, four backends)

## Architecture invariants

These are the load-bearing design choices. Preserve them when changing code.

**Castra is game-agnostic.** [foedus](https://github.com/dbyrne/foedus) is the first user, but castra **must not import** game-specific code or assume Diplomacy-shaped semantics. The integration boundary is the `WorkUnit` Protocol (`castra/protocol.py`) — three methods (`from_dict`, `to_dict`, `run`). User projects register `WorkUnit` subtypes by name; the coordinator stores shards as `(type_name, payload)` and the worker dispatches via the registry. **Don't add foedus-specific anything to castra.**

**One worker process implementation, four spawn methods.** `castra worker` is the same Python entry point whether spawned local-inproc, local-subprocess, over SSH, or inside Docker on EC2. Only the *spawn command* differs (`fleet.py` per backend). Don't fork the worker per backend — fork the spawner.

**The coordinator is the only durable-state owner.** Workers write nothing durable; they claim, run, report. State lives in `runs.db` (SQLite) + per-run JSONL event logs (`runs/<run-id>.jsonl`). Lease expiry (`claimed → pending` with attempt count incremented) is the *only* recovery path for crashed workers — don't add worker-side persistence.

**Worktree-per-experiment is the spine.** Each `castra exp create <name>` creates a sibling-of-main worktree at `<project-root>-<name>/` on branch `experiment/<name>`. Inside the worktree, runtime state lives at `experiments/<name>/` (config, checkpoints, runs, logs). Inside the *main* repo, `experiments/<name>/` only contains state that's been `ship`-ped back. **Don't blur this boundary.** Working state lives in the worktree, archived state in main.

**`fork` inherits both code (via branch) and checkpoint (via resolution policy).** `castra exp fork foo --from bar --checkpoint latest|finalized|auto` is the killer feature: try a variation in one command. The new worktree branches off `experiment/<src>` if it exists, else `main`. Don't simplify away the parent-checkpoint resolution.

**Validation is strict.** ExperimentSpec rejects unknown keys. Templates live in `castra/templates/*.yaml` (bundled as package data — see `pyproject.toml` `[tool.setuptools.package-data]`). Dotted-key overrides at create time: `castra exp create foo -t base -o axes.nn.training.hyperparams.lr=1e-4`.

## Docker glue (`castra/image.py`)

Shells out to the `docker` CLI rather than using the Python `docker` SDK — same rationale as foedus's agent_build.py:

- Python `docker` SDK has a brittle dependency tree (paramiko, urllib3 pinning).
- Most environments running castra already have the CLI.
- Swapping `podman` later is one command-name change.

Registry-agnostic by design (dockerhub / GHCR / ECR / local). ECR-specific multi-region push and IAM glue lands when the EC2 backend ships (Phase 6+ in `DESIGN.md`).

## Module → test mapping

The test suite is one-test-file-per-module. When you add `castra/foo.py`, add `tests/test_foo.py`. When changing module N's behavior, run `pytest tests/test_N.py` first; only run the full suite once the local module is green.

| Module | Purpose |
|---|---|
| `protocol.py` | `WorkUnit` Protocol + name registry — **the only user integration point** |
| `spec.py` | `ExperimentSpec` dataclass + YAML I/O + dotted-key overrides |
| `paths.py` | Worktree path conventions (sibling-of-main, branch naming) |
| `git.py` / `worktree.py` | Worktree lifecycle (create / fork / remove); branches retained on remove |
| `venv.py` | Idempotent venv setup (system-site-packages) |
| `commands.py` | Implementation of CLI verbs (`cmd_exp_*`) — keep `cli.py` thin |
| `coordinator.py` | fastapi app: `/workers/*`, `/shards/*`, `/experiments/*` |
| `client.py` | `CoordinatorClient` — workers' HTTP wrapper |
| `worker.py` | Single worker implementation (claim → deserialize → run → report) |
| `storage.py` | SQLite + JSONL persistence (`Store`, `ShardRecord`) |
| `fleet.py` | Backend spawners: local-inproc, local-subprocess, ssh, ec2-* |
| `capacity.py` | Capacity YAML → allocation plan (cost-ordered fill) |
| `image.py` | docker CLI shellout; registry-agnostic |
| `dashboard.py` | Rich-based TUI (3-pane: experiments × detail × fleet) |

## Non-goals (per `DESIGN.md`)

When tempted to add: hyperparameter optimization algorithms (Bayesian, ASHA), model serving, a learned scheduler, generic workflow-engine features (Airflow/Prefect/Temporal), multi-tenant auth, alternative-VCS support — **don't**. Each of these is explicitly out of scope. If a user wants them, they compose castra with another tool. Castra stays small.
