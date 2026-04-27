"""ExperimentSpec dataclass + YAML I/O + dotted overrides."""

from __future__ import annotations

import pytest

from castra.spec import (
    ExperimentSpec,
    apply_override,
    apply_overrides,
    parse_value,
)


def test_parse_value_int() -> None:
    assert parse_value("42") == 42


def test_parse_value_float() -> None:
    assert parse_value("1.5") == 1.5


def test_parse_value_scientific() -> None:
    assert parse_value("1e-4") == 0.0001


def test_parse_value_bool_true() -> None:
    assert parse_value("true") is True


def test_parse_value_bool_false() -> None:
    assert parse_value("false") is False


def test_parse_value_null() -> None:
    assert parse_value("null") is None


def test_parse_value_string() -> None:
    assert parse_value("hello") == "hello"


def test_apply_override_creates_intermediates() -> None:
    d: dict = {}
    apply_override(d, "axes.nn.training.lr", 0.001)
    assert d == {"axes": {"nn": {"training": {"lr": 0.001}}}}


def test_apply_override_modifies_existing() -> None:
    d = {"axes": {"nn": {"training": {"lr": 0.1}}}}
    apply_override(d, "axes.nn.training.lr", 0.0001)
    assert d["axes"]["nn"]["training"]["lr"] == 0.0001


def test_apply_override_through_non_dict_raises() -> None:
    d = {"axes": "not-a-dict"}
    with pytest.raises(ValueError):
        apply_override(d, "axes.nn", 1)


def test_apply_overrides_list() -> None:
    d: dict = {}
    apply_overrides(d, ["axes.nn.training.lr=1e-4", "env.num_players=4"])
    assert d["axes"]["nn"]["training"]["lr"] == 0.0001
    assert d["env"]["num_players"] == 4


def test_apply_overrides_bad_format() -> None:
    with pytest.raises(ValueError):
        apply_overrides({}, ["no-equals-sign"])


def test_spec_from_dict_strict_validation() -> None:
    """Unknown top-level keys are rejected."""
    with pytest.raises(ValueError):
        ExperimentSpec.from_dict({"name": "x", "totally_unknown": 1})


def test_spec_round_trip(tmp_path) -> None:
    spec = ExperimentSpec(
        name="exp-1",
        parent="exp-0",
        axes={"nn": {"hidden_dim": 256}},
        env={"num_players": 4},
    )
    path = tmp_path / "config.yaml"
    spec.to_yaml(path)
    loaded = ExperimentSpec.from_yaml(path)
    assert loaded.name == "exp-1"
    assert loaded.parent == "exp-0"
    assert loaded.axes == {"nn": {"hidden_dim": 256}}
    assert loaded.env == {"num_players": 4}


def test_spec_defaults() -> None:
    spec = ExperimentSpec(name="x")
    assert spec.parent is None
    assert spec.axes == {}
    assert spec.concluded_gen is None
