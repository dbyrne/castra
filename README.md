# castra

Experiment management and worker fleet for AI agent research.

Where the legions train.

## What it is

`castra` (Latin: *fortified military camp*) is a CLI for managing the lifecycle of agent-research experiments — from spec definition through worktree-isolated training to multi-machine evaluation. It handles the boring-but-load-bearing parts of agent research:

- **Two orthogonal experiment axes** as first-class — e.g. NN architecture × LLM fine-tune — so you can vary one while pinning the other.
- **Git worktree per experiment**, with `fork` from a parent that inherits checkpoints. Run twenty variants in parallel without stepping on yourself.
- **Single worker protocol over four spawn backends**: local in-process, local subprocess, SSH (other machines you own), and EC2 (spot or on-demand). Same Python, same wire protocol — only the `spawn` command differs.
- **Coordinator** with shard claim / complete / fail / heartbeat semantics. Workers come and go; shards are leased and returned to the queue on lease expiry.
- **Static Docker images keyed to git SHA**, multi-region ECR push, EC2 launch-plan generation with capacity config.
- **Tag-based EC2 cost tracking** — Spot price API + configured fallbacks, recorded snapshots.
- **Dual-layer run logs** — SQLite for fast queries, JSONL event logs for full replay.
- **TUI dashboard** showing experiments × workers × cost in one view.

You implement a small `WorkUnit` Protocol for your project; castra handles the rest.

## Why this exists separately

Each agent builder ends up writing some version of this tooling — worktree management, worker spawning, run tracking, cost accounting — and each version is opinionated to one project. `castra` extracts the parts that are not opinionated to any specific game/domain, so the next agent project doesn't have to start from scratch.

The first user is [foedus](https://github.com/dbyrne/foedus), but castra is intentionally game-agnostic.

## Status

**Design phase.** No code yet. See [`DESIGN.md`](DESIGN.md) for the architecture.

The first milestone (worktree lifecycle + local backends + coordinator) targets ~1–2 weeks of work. EC2 / Docker / multi-machine support layers on after.

## License

MIT — see [`LICENSE`](LICENSE).
