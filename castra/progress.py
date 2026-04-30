"""Real-time progress for long-running experiments.

User projects (training scripts, eval pipelines, sweep harnesses) call
`update_progress()` at stage transitions to publish status updates;
the castra dashboard reads them at 1 Hz and renders stage / message
columns next to each experiment.

File-based — no coordinator dependency, survives crashes, and works
for non-distributed scripts. Single-writer (the orchestrator) /
many-readers (dashboard, monitoring tools) is safe via atomic
`os.replace`.

Layout: `<experiment_dir>/progress.json` next to `config.yaml`. When
`castra exp ship` copies `experiments/<name>/` back to main, progress
ships along with it (showing the final status). `castra exp archive`
removes it along with the worktree.

Schema:
  stage:       short label, e.g. "self-play", "train policy", "gate"
  message:     human-readable detail, e.g. "32/100 shards done"
  updated_at:  ISO 8601 UTC timestamp
  extra:       optional dict of structured metrics (numeric or string)
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROGRESS_FILENAME = "progress.json"

# Progress entries older than this are flagged as "stalled" — useful for
# spotting hung runs in the dashboard.
STALE_AFTER_SECONDS = 300.0


@dataclass
class Progress:
    stage: str
    message: str = ""
    updated_at: str = ""
    extra: dict[str, Any] | None = None

    @property
    def updated_dt(self) -> _dt.datetime | None:
        if not self.updated_at:
            return None
        try:
            return _dt.datetime.fromisoformat(
                self.updated_at.replace("Z", "+00:00")
            )
        except (TypeError, ValueError):
            return None

    @property
    def age_seconds(self) -> float | None:
        dt = self.updated_dt
        if dt is None:
            return None
        return (_dt.datetime.now(_dt.timezone.utc) - dt).total_seconds()

    @property
    def is_stale(self) -> bool:
        age = self.age_seconds
        return age is not None and age > STALE_AFTER_SECONDS


def update_progress(
    experiment_dir: Path | str,
    *,
    stage: str,
    message: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically write a progress update to `experiment_dir/progress.json`.

    Atomic via temp + os.replace so dashboard polls never see a half-written
    file. Failure to write is non-fatal — the progress file is a hint, not
    durable state.
    """
    exp_dir = Path(experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "stage": stage,
        "message": message,
        "updated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    if extra:
        payload["extra"] = extra
    path = exp_dir / PROGRESS_FILENAME
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass


def read_progress(experiment_dir: Path | str) -> Progress | None:
    """Read `experiment_dir/progress.json`. Returns None if missing or
    unreadable (defensive — readers shouldn't crash on a bad file)."""
    path = Path(experiment_dir) / PROGRESS_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return Progress(
        stage=str(data.get("stage", "")),
        message=str(data.get("message", "")),
        updated_at=str(data.get("updated_at", "")),
        extra=data.get("extra") if isinstance(data.get("extra"), dict) else None,
    )


def clear_progress(experiment_dir: Path | str) -> None:
    """Remove the progress.json file (e.g. on experiment completion)."""
    path = Path(experiment_dir) / PROGRESS_FILENAME
    try:
        path.unlink()
    except FileNotFoundError:
        pass
