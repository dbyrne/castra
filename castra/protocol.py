"""The WorkUnit Protocol — castra's only integration point for user projects.

User projects define classes implementing the protocol and register them by
name. The coordinator stores each shard as `(type_name, payload)`; workers
look up the registered class and reconstruct the WorkUnit before calling
`run()`.

Example (in a hypothetical user project):

    from castra.protocol import register

    @register("TournamentShard")
    class TournamentShard:
        def __init__(self, agent_specs, seed):
            self.agent_specs = agent_specs
            self.seed = seed

        @classmethod
        def from_dict(cls, data):
            return cls(agent_specs=data["agents"], seed=data["seed"])

        def to_dict(self):
            return {"agents": self.agent_specs, "seed": self.seed}

        def run(self):
            # play one game, return a JSON-serializable result dict
            return {"winner": ..., "scores": ...}
"""

from __future__ import annotations

import threading
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WorkUnit(Protocol):
    """A serializable unit of work executed by a worker."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkUnit": ...

    def to_dict(self) -> dict[str, Any]: ...

    def run(self) -> dict[str, Any]:
        """Execute and return a JSON-serializable result dict."""
        ...


_REGISTRY: dict[str, type] = {}
_LOCK = threading.Lock()


def register(type_name: str):
    """Decorator to register a `WorkUnit` class under a stable name.

    The same class re-registered under the same name is a no-op (so module
    re-imports are safe). Registering a different class under the same name
    raises ValueError.
    """

    def deco(cls: type) -> type:
        with _LOCK:
            existing = _REGISTRY.get(type_name)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"WorkUnit type {type_name!r} already registered to {existing!r}"
                )
            _REGISTRY[type_name] = cls
        return cls

    return deco


def get(type_name: str) -> type:
    """Look up a registered class by name. Raises KeyError if missing."""
    cls = _REGISTRY.get(type_name)
    if cls is None:
        raise KeyError(
            f"WorkUnit type {type_name!r} not registered. "
            f"Did you forget to import the module that defines it?"
        )
    return cls


def deserialize(type_name: str, payload: dict[str, Any]) -> WorkUnit:
    """Look up a registered class and reconstruct an instance from a payload."""
    return get(type_name).from_dict(payload)


def list_registered() -> list[str]:
    with _LOCK:
        return sorted(_REGISTRY.keys())


def clear_registry() -> None:
    """Test helper. Do not use in production code."""
    with _LOCK:
        _REGISTRY.clear()
