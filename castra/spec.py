"""ExperimentSpec — typed dataclass with YAML I/O and dotted-key overrides.

The top-level shape is fixed; the contents of `axes`, `env`, `shards`, and
`artifacts` are opaque dicts (user projects validate their own contents).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentSpec:
    """The canonical definition of an experiment.

    `axes` is a dict of axis-name -> opaque-dict. Canonical axis names are
    `nn` (tactics) and `llm` (negotiation), but castra doesn't enforce them
    — user projects pick what makes sense for their domain.
    """

    name: str
    parent: str | None = None
    parent_checkpoint: str | None = None  # null | latest | finalized | gen<N>

    axes: dict[str, dict[str, Any]] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    shards: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    # Lifecycle metadata, set by castra (not by user).
    created_at: str | None = None
    git_sha: str | None = None
    concluded_gen: int | None = None
    concluded_reason: str | None = None
    concluded_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentSpec":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown spec keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentSpec":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def parse_value(s: str) -> Any:
    """Parse a CLI override value with sensible type inference.

    Tries (in order): explicit-quoted string, int, float, then YAML for
    bool / null / unquoted strings. The float try is what catches forms
    like `1e-4`, which YAML 1.1 rejects as scalar (requires explicit sign).
    """
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return yaml.safe_load(s)


def apply_override(spec_dict: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Apply `axes.nn.training.lr = 1e-4` style override in place.

    Creates intermediate dicts as needed. Raises ValueError if a path
    traverses a non-dict value.
    """
    parts = dotted_key.split(".")
    target = spec_dict
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        if not isinstance(target[part], dict):
            raise ValueError(
                f"cannot apply override {dotted_key!r}: "
                f"path traverses non-dict at {part!r}"
            )
        target = target[part]
    target[parts[-1]] = value


def apply_overrides(
    spec_dict: dict[str, Any], overrides: list[str]
) -> dict[str, Any]:
    """Apply CLI overrides of the form 'dotted.key=value' to a spec dict.

    Returns the modified dict (also modifies in place).
    """
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"override must be 'key=value', got {raw!r}")
        key, value_str = raw.split("=", 1)
        apply_override(spec_dict, key.strip(), parse_value(value_str.strip()))
    return spec_dict
