"""Tests for the WorkUnit Protocol and registry."""

from __future__ import annotations

from typing import Any

import pytest

from castra import protocol


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test gets a clean registry."""
    protocol.clear_registry()
    yield
    protocol.clear_registry()


class _Sample:
    """A tiny test WorkUnit."""

    def __init__(self, value: int) -> None:
        self.value = value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_Sample":
        return cls(value=data["value"])

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    def run(self) -> dict[str, Any]:
        return {"doubled": self.value * 2}


def test_register_and_get() -> None:
    @protocol.register("Sample")
    class S(_Sample):
        pass

    assert protocol.get("Sample") is S


def test_register_idempotent_same_class() -> None:
    """Re-registering the same class under the same name is a no-op."""

    @protocol.register("Sample")
    class S(_Sample):
        pass

    # Re-register the same class (e.g. module reimport)
    protocol.register("Sample")(S)
    assert protocol.get("Sample") is S


def test_register_conflict_raises() -> None:
    @protocol.register("Sample")
    class A(_Sample):
        pass

    with pytest.raises(ValueError):
        @protocol.register("Sample")
        class B(_Sample):
            pass


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError):
        protocol.get("Ghost")


def test_deserialize_roundtrip() -> None:
    @protocol.register("Sample")
    class S(_Sample):
        pass

    obj = protocol.deserialize("Sample", {"value": 21})
    assert isinstance(obj, S)
    assert obj.run() == {"doubled": 42}


def test_list_registered() -> None:
    @protocol.register("A")
    class A(_Sample):
        pass

    @protocol.register("B")
    class B(_Sample):
        pass

    assert protocol.list_registered() == ["A", "B"]
