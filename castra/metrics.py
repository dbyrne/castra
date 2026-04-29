"""MetricsRecord — standardized benchmark output for `castra exp compare`.

Castra is game-agnostic; this schema is deliberately opaque to castra. User
projects (auspex, foedus, …) emit a `metrics.yaml` after a benchmark run;
`castra exp compare` reads them and tabulates side-by-side. Castra cares
only about the shape (`metrics: dict[str, number]` + `metadata: dict`),
not what individual keys mean.

Layout: `experiments/<name>/benchmarks/metrics.yaml` inside the worktree
during a run; `cmd_exp_ship` copies the whole `benchmarks/` directory back
to the main repo when the experiment concludes.

Keys in `metrics` are expected to be numeric (int or float) so that
`compare` can compute deltas. Non-numeric values land in `metadata` and
are shown as plain strings without delta columns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


METRICS_FILENAME = "metrics.yaml"


@dataclass
class MetricsRecord:
    """One benchmark result.

    `metrics`: numeric key→value pairs. Castra treats missing keys
    across experiments as "—" rather than failing.

    `metadata`: free-form scalars (timestamp, tool name, search mode, …).
    Compared as-is, no delta column.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricsRecord":
        if not isinstance(data, dict):
            raise ValueError(f"metrics record must be a dict, got {type(data).__name__}")
        known = {"metrics", "metadata"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown metrics keys: {sorted(unknown)}")

        raw_metrics = data.get("metrics") or {}
        if not isinstance(raw_metrics, dict):
            raise ValueError("`metrics` must be a dict")
        # Coerce all values to floats. Non-numeric here is a usage error —
        # put it in `metadata` instead.
        metrics: dict[str, float] = {}
        for k, v in raw_metrics.items():
            if isinstance(v, bool):
                # bool is an int subclass; reject so users put it in metadata.
                raise ValueError(
                    f"metric {k!r}=bool — put booleans in `metadata` instead."
                )
            if not isinstance(v, (int, float)):
                raise ValueError(
                    f"metric {k!r}={v!r} is not numeric — put non-numeric "
                    "values in `metadata` instead."
                )
            metrics[k] = float(v)

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("`metadata` must be a dict")

        return cls(metrics=metrics, metadata=dict(metadata))

    @classmethod
    def from_yaml(cls, path: Path) -> "MetricsRecord":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def load_metrics(experiment_dir: Path) -> MetricsRecord | None:
    """Load `<experiment_dir>/benchmarks/metrics.yaml` if present.

    `experiment_dir` is the directory that contains `config.yaml` and
    `benchmarks/` — typically `<worktree>/experiments/<name>/` or, for
    shipped experiments, `<project_root>/experiments/<name>/`.

    Returns None if the file isn't present.
    """
    path = experiment_dir / "benchmarks" / METRICS_FILENAME
    if not path.exists():
        return None
    return MetricsRecord.from_yaml(path)
