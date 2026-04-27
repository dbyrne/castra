# castra — design

Status: design phase, no code yet. This document is the architectural intent.

## Goals

1. **Two orthogonal experiment axes are first-class.** A canonical example: a strategic neural network is one axis, a fine-tuned LLM negotiator is another. Specs, fork, eval, and dashboard all treat the two axes symmetrically. Either may be null for a single-axis experiment.
2. **One worker process implementation, four spawn methods.** Local in-process, local subprocess, SSH, EC2. Same wire protocol, same Python entry point. Only the *how the process gets started* differs.
3. **Worktree-per-experiment with a real `fork` mechanism.** Each experiment gets its own physical directory and branch. Forking from a parent inherits the parent's code changes (via the branch) and checkpoint (via a resolution policy).
4. **Reproducibility without ceremony.** Every result tied to a git SHA and a config hash. Worker images are keyed to git SHA; shard manifests assert worker compatibility.
5. **Costs visible.** EC2 spend tracked by tag, recorded as snapshots; local and SSH backends report zero cost so the same `cost` command works everywhere.
6. **Stays small.** The framework should not dwarf the projects that use it.

## Glossary

- **Experiment** — a concrete configuration of training/eval, defined by a YAML spec.
- **Worktree** — a git worktree containing the code and runtime state for one experiment.
- **Spec** — the YAML/dataclass definition of what an experiment is.
- **Shard** — a discrete unit of work (one self-play game, one tournament match, one gradient batch).
- **Coordinator** — a fastapi sidecar that distributes shards to workers and aggregates results.
- **Worker** — a process that polls the coordinator for shards and runs them.
- **Backend** — a method for spawning workers (`local-inproc`, `local-subprocess`, `ssh`, `ec2-*`).
- **WorkUnit** — a user-defined Python class implementing `run() -> dict`. The integration point for castra-using projects.
- **Frontier** — the currently-promoted "champion" checkpoint for a project.

## Architecture overview

```
+---------------+        spawn         +-----------+
|   castra cli  | -------------------> |  worker   |  one process,
|  (main repo)  |                      |  process  |  four spawn methods
+---------------+        ^             +-----------+
        |                |                  |
        |                |  registers       | claims shards
        v                |                  v
+---------------+    +---------------+    +-------------+
|   worktrees   |    |  coordinator  | <- |  shard      |
|  (git, venv,  |    |  (fastapi)    |    |  queue      |
|   ckpts)      |    +---------------+    +-------------+
+---------------+         |
        ^                 v
        |          +---------------+
        +----------|  runs.db      |
                   |  + JSONL logs |
                   +---------------+
```

The user-facing CLI (`castra`) drives everything. Workers know only the coordinator URL; they don't care what shards are *for*. The coordinator owns durable state (SQLite + JSONL).

## Integration: the WorkUnit Protocol

This is the only contract user projects implement.

```python
# castra/protocol.py
from typing import Protocol

class WorkUnit(Protocol):
    """A serializable unit of work."""

    @classmethod
    def from_dict(cls, data: dict) -> "WorkUnit": ...

    def to_dict(self) -> dict: ...

    def run(self) -> dict:
        """Execute the work and return a JSON-serializable result."""
```

A project registers its `WorkUnit` subtypes by name. The coordinator stores shards as `(type_name, payload)`; the worker dispatches on `type_name` to the right class.

Example user implementation (in a project, not castra itself):

```python
# Hypothetical example in a user project
class TournamentShard:
    def __init__(self, agent_specs, game_config, seed):
        self.agent_specs = agent_specs
        self.game_config = game_config
        self.seed = seed

    @classmethod
    def from_dict(cls, data): ...
    def to_dict(self): ...
    def run(self) -> dict:
        # play one game, return result
        ...
```

User projects do not import the bulk of castra — they import the protocol and a few helpers. Castra has no game-specific knowledge.

## ExperimentSpec (YAML)

Two orthogonal axes (`a` and `b`) plus generic metadata. Names of axes are user-defined; canonical examples are `nn` and `llm` but castra doesn't enforce this.

```yaml
name: nn-v3-with-qwen-tuned-v2
parent: nn-v3                                # null = fresh; name = fork
parent_checkpoint: latest                    # null | latest | finalized | gen<N>

axes:
  nn:                                        # may be null
    architecture: { type: mlp_attention, hidden_dim: 256, ... }
    training:    { algorithm: ppo, hyperparams: {...}, schedule: {...} }
  llm:                                       # may be null
    base_model: qwen2.5:7b
    fine_tune:  { adapter: ..., method: lora }
    prompt_template: prompts/negotiator-v2.j2
    bias_mechanism: constraints

env:                                          # passed to user project
  num_players: 4
  peace_threshold: 5

shards:
  selfplay:    { count: 1_000_000, batch_size: 1024 }
  tournament:  { games: 200, pool: configs/champion-pool.yaml }
  scenarios:   { suite: scenarios/regression-v1.yaml }

artifacts: { checkpoints_keep: 10, retain_finalized: true }
```

Templates live in `experiments/_templates/`. Dotted-key overrides at create time: `castra exp create foo -t base -o axes.nn.training.hyperparams.lr=1e-4`.

The schema is a Python dataclass. Validation is strict — unknown keys are an error, not a warning.

## Worktree lifecycle

Directory layout from `castra exp create my-experiment`:

```
project/                              # main repo (where you run castra)
  experiments/<name>/                 # synced from worktree on `ship`:
    config.yaml                       #   final config
    benchmarks/                       #   final eval results
    checkpoints/                      #   only finalized + concluded ckpt
  frontier/                           # current champion (post-`promote`)

../project-<name>/                    # worktree on branch experiment/<name>
  <project files at branch HEAD>
  experiments/<name>/                 # working dir, NOT in main
    config.yaml
    checkpoints/{latest.pt, gen-NNNN.pt}
    runs.db                           # SQLite
    runs/<run-id>.jsonl               # per-shard event logs
    benchmarks/{tournament,scenarios}.jsonl
    logs/
  .venv/                              # --system-site-packages, idempotent setup
```

Verbs:

| Command | Effect |
|---|---|
| `castra exp create <name> [-t template] [-o k=v ...]` | git worktree + branch + venv + initial config |
| `castra exp fork <name> --from <src> [--checkpoint auto\|latest\|finalized\|genN]` | new worktree branched off `experiment/<src>` if it exists else main; inherit parent ckpt |
| `castra exp repair <name>` | idempotent venv re-setup (no destructive ops) |
| `castra exp finalize <name> --gen N --reason "..."` | mark canonical conclusion; writes config.concluded_gen + ts |
| `castra exp ship <name>` | sync config + benchmarks back to main; code merge is separate |
| `castra exp promote <name> [<gen>]` | copy ckpt to `frontier/`; update `FRONTIER.md` with pre/post scores |
| `castra exp archive <name>` | rm worktree + venv (branch and concluded artifacts retained) |
| `castra exp list` / `info` / `compare <names...>` / `dashboard` | queries |

The fork mechanism is the killer feature. "Try a variation" becomes one command, with parent code and parent checkpoint inherited atomically.

## Fleet — one protocol, four backends

### Worker process

Single Python implementation:

```
castra worker --coordinator <url> [--threads N] [--lease-seconds 240]
```

Polls coordinator, claims a shard, dispatches by `type_name` to the registered `WorkUnit` class, calls `.run()`, posts the result, repeats. Renews lease on heartbeat. Writes nothing durable.

### Coordinator

```
castra coordinator start <experiment> [--port 8765] [--bind 0.0.0.0]
```

A fastapi sidecar. State in `runs.db` (SQLite).

| Endpoint | Purpose |
|---|---|
| `POST /workers/register` | Worker announces, gets ID |
| `POST /workers/<id>/heartbeat` | Lease renewal |
| `POST /shards/claim` | Long-poll for next shard |
| `POST /shards/<id>/complete` | Submit result |
| `POST /shards/<id>/fail` | Return shard to queue |
| `GET /experiments/<name>/status` | Snapshot for TUI/dashboard |
| `GET /experiments/<name>/replay/<run-id>` | Stream JSONL event log |

Shard state: `pending → claimed → completed | failed`. Lease expiry returns `claimed → pending` (with attempt count incremented).

### Backends

| Backend | Spawn command | Notes |
|---|---|---|
| `local-inproc` | (none — coordinator runs the work itself) | Dev only, no parallelism |
| `local-subprocess` | `python -m castra.worker --coordinator http://localhost:8765` | One per CPU core typically |
| `ssh` | `ssh <host> "<venv>/bin/python -m castra.worker --coordinator <master-url>"` | Master URL via Tailscale or LAN |
| `ec2-spot` / `ec2-ondemand` | `aws ec2 run-instances` with cloud-init that runs Docker | Requires `worker-image build` first |

### Capacity config

A single YAML describes available compute:

```yaml
local:
  enabled: true
  workers: auto                                # = num_cpu - 1
  threads_per_worker: 1

ssh:
  reachable_via: tailscale                     # how the master URL works
  hosts:
    - { host: laptop-living-room, user: david, ssh_key: ~/.ssh/id_ed25519, workers: 4 }
    - { host: laptop-basement,    user: david, workers: 8 }

ec2:
  regions: [us-east-1, us-west-2]
  instance_types: [c7i.4xlarge, c7i.8xlarge]
  market: spot                                 # spot | on-demand | mixed
  ami: ami-...
  subnet_ids: [...]
  security_group_ids: [...]
  iam_instance_profile: castra-worker
  hourly_prices:
    spot:
      us-east-1: { c7i.4xlarge: 0.50 }
  threads_per_worker: 1
  worker_count: auto
```

`castra workers plan --capacity capacity.yaml --max N` outputs allocation ordered by cost (local zero-cost first, then SSH, then EC2). `castra workers launch` actually spawns. `castra workers cost` is generic — local/SSH report zero, EC2 queries by tag and computes from Spot API + configured fallbacks.

### Worker image (for EC2)

`castra worker-image build <name> --push --regions us-east-1,us-west-2` runs:

```
docker build --build-arg CASTRA_GIT_SHA=<sha> -f Dockerfile.worker -t <repo>:<exp>-<sha> .
aws ecr get-login-password | docker login ...
docker push <repo>:<exp>-<sha>            # for each region
```

Records the build to `worker_images.jsonl` with a code fingerprint computed from a curated list of paths. Per-gen compatibility is enforced via `required_worker_fingerprint` written into shard manifests — workers refuse to claim shards their fingerprint doesn't match.

## Storage

Per-experiment SQLite + JSONL.

```sql
CREATE TABLE shards (
  shard_id TEXT PRIMARY KEY,
  experiment TEXT,
  type_name TEXT,
  payload_json TEXT,
  status TEXT,                     -- pending | claimed | completed | failed
  worker_id TEXT,
  attempts INTEGER DEFAULT 0,
  claimed_at TEXT,
  lease_expires_at TEXT,
  completed_at TEXT,
  result_json TEXT
);

CREATE TABLE workers (
  worker_id TEXT PRIMARY KEY,
  backend TEXT,                    -- local-subprocess | ssh | ec2-spot | ...
  hostname TEXT,
  ec2_instance_id TEXT,
  registered_at TEXT,
  last_heartbeat_at TEXT,
  status TEXT
);

CREATE TABLE runs (
  run_id TEXT PRIMARY KEY,
  shard_id TEXT,
  experiment TEXT,
  started_at TEXT,
  ended_at TEXT,
  outcome TEXT,
  result_json TEXT
);
```

Per-run event logs in `runs/<run-id>.jsonl`. Each line: `{ts, event_type, detail}`. Replay by streaming the file.

## CLI surface

```
# Lifecycle
castra exp create <name> [-t template] [-o k=v ...]
castra exp fork   <name> --from <src> [--checkpoint <spec>]
castra exp repair <name>
castra exp finalize <name> --gen N --reason "..."
castra exp ship   <name>
castra exp archive <name>
castra exp promote <name> [<gen>]

# Queries
castra exp list / info / compare / dashboard

# Coordinator + workers
castra coordinator start / status / stop  <name>
castra workers plan / launch / cost / list  <name>
castra worker-image build / list  <name>

# The single worker (anywhere)
castra worker --coordinator <url> [--threads N]
```

User projects add their own action commands (`my-project train`, `my-project tournament`, etc.) — those are not part of castra's CLI. Castra provides the lifecycle + fleet plumbing; project-specific commands live in the project.

## TUI

`castra exp dashboard` — three-pane Rich-based TUI:

- **Left**: experiments sorted *running → finalized → stopped*. Each row: name, axis-A version, axis-B version, current rating/metric, phase.
- **Top right**: selected experiment detail — recent shard outcomes, eval scores, last log lines.
- **Bottom right**: fleet view — workers grouped by backend (`local: 8`, `ssh: 12`, `ec2-spot: 20`), claimed/completed shard counts, $/hr estimate.

Polls coordinator at 1 Hz. Target: ~250 LOC.

A web dashboard is deferred. The TUI is sufficient for a single user iterating; the web becomes worth building when multiple humans need to watch.

## Build phases

Order optimized for "shortest path to actually running an experiment end-to-end":

**Phase 1 — Worktree lifecycle, no workers.**
`exp create / fork / repair / list / info`. Disciplines checkpoint hygiene immediately and gives users something to commit to.

**Phase 2 — Coordinator + `local-inproc` backend.**
Coordinator runs the work itself when zero workers are registered. Validates the wire protocol without parallelism.

**Phase 3 — `local-subprocess` worker.**
Now a project can use all CPU cores. First real fleet.

**Phase 4 — TUI dashboard.**
Useful any time after Phase 3.

**Phase 5 — `ssh` backend.**
~100 LOC for the spawn shim plus a Tailscale/LAN reachability check. Adds laptop-as-fleet-member.

**Phase 6 — Worker image build + ECR push.**
Static images keyed to git SHA. Multi-region.

**Phase 7 — `ec2-spot` / `ec2-ondemand` backends + cost tracking.**
Capacity config, launch plan, tag-based discovery, Spot pricing.

**Phase 8 — `compare` and `promote` polish.**
Statistical comparisons (rating-delta CIs), Frontier promotion workflow with pre/post-eval scores, drift detection.

Phases 1–4 are the MVP. After Phase 4, a project can run real experiments at single-machine scale. Phases 5–7 add multi-machine; Phase 8 adds the discipline-enforcing polish.

## Non-goals

- **Hyperparameter optimization algorithms** (Bayesian opt, ASHA, etc.). Sweeps are a *spec product* (cartesian or sampled), not a search algorithm. If users want fancy search, they generate spec lists with their own code.
- **Model serving / inference deployment.** Castra's domain is *training and evaluation*, not production inference.
- **A learned scheduler.** Capacity allocation is rule-based (cost-ordered fill). No RL meta-controller.
- **Generic workflow engine** (Airflow / Prefect / Temporal). Castra is specifically about agent-research lifecycles. If you need DAGs of arbitrary tasks, use one of those.
- **Multi-tenant / multi-user.** Single user, single project per coordinator. Multi-tenancy can be added later if needed; it changes the security model materially.
- **Alternative VCS support.** git only. Worktree semantics are the spine of the design.

## Open questions

These are deliberate uncertainties, to be resolved by usage rather than upfront speculation:

1. **Spec composition.** Should specs support `extends:` for inheritance from another spec, beyond fork? Maybe in Phase 8.
2. **Streaming results.** Do we need the coordinator to push partial results during a long shard, or is "report once on completion" enough? Defaulting to the latter; revisit if a project needs progress bars.
3. **Multi-coordinator sharding.** When does a single coordinator stop scaling? Probably when shard volume exceeds a few thousand per minute. Cross that bridge when we get there.
4. **Auth.** Coordinator is currently unauthenticated (LAN/Tailscale assumed). What's the right minimum auth for a public-internet coordinator? TBD before EC2 backend ships.
