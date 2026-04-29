"""MetricsRecord schema + load_metrics."""

from __future__ import annotations

import yaml
import pytest

from castra.metrics import METRICS_FILENAME, MetricsRecord, load_metrics


# --- Round-trip --------------------------------------------------------------


def test_round_trips_through_yaml(tmp_path):
    """Write + read = identity."""
    rec = MetricsRecord(
        metrics={"delta": 3.71, "advisor_wins": 79, "ties": 102},
        metadata={"tool": "compare_advisor", "search": "puct",
                  "timestamp": "2026-04-29T20:30:00Z"},
    )
    path = tmp_path / METRICS_FILENAME
    rec.to_yaml(path)
    revived = MetricsRecord.from_yaml(path)
    # Numerics get coerced to float, so compare via dict.
    assert revived.metadata == rec.metadata
    assert revived.metrics.keys() == rec.metrics.keys()
    for k in rec.metrics:
        assert revived.metrics[k] == pytest.approx(rec.metrics[k])


def test_empty_record_is_valid(tmp_path):
    rec = MetricsRecord()
    path = tmp_path / "m.yaml"
    rec.to_yaml(path)
    revived = MetricsRecord.from_yaml(path)
    assert revived.metrics == {}
    assert revived.metadata == {}


def test_int_values_become_floats():
    """Numeric metrics are stored as floats so castra compare can compute deltas
    uniformly. Values from YAML come back as float regardless of authored type."""
    rec = MetricsRecord.from_dict({
        "metrics": {"games": 200, "delta": 3.71},
    })
    assert isinstance(rec.metrics["games"], float)
    assert rec.metrics["games"] == 200.0


# --- Validation --------------------------------------------------------------


def test_rejects_unknown_top_level_keys():
    with pytest.raises(ValueError, match="unknown metrics keys"):
        MetricsRecord.from_dict({
            "metrics": {"a": 1.0},
            "results": {"oops": 1},
        })


def test_rejects_non_numeric_metric():
    with pytest.raises(ValueError, match="not numeric"):
        MetricsRecord.from_dict({"metrics": {"label": "puct"}})


def test_rejects_bool_metric():
    """Booleans are int-subclass so we reject them explicitly. They belong
    in `metadata`."""
    with pytest.raises(ValueError, match="bool"):
        MetricsRecord.from_dict({"metrics": {"converged": True}})


def test_rejects_non_dict_metrics_value():
    with pytest.raises(ValueError, match="`metrics` must be a dict"):
        MetricsRecord.from_dict({"metrics": [1, 2, 3]})


def test_metadata_allows_arbitrary_scalars():
    rec = MetricsRecord.from_dict({
        "metadata": {
            "search": "puct",
            "epochs": 10,
            "ckpt": None,
        },
    })
    assert rec.metadata["search"] == "puct"
    assert rec.metadata["epochs"] == 10
    assert rec.metadata["ckpt"] is None


# --- load_metrics ------------------------------------------------------------


def test_load_metrics_returns_none_when_missing(tmp_path):
    """Missing benchmarks/metrics.yaml is not an error — returns None."""
    exp_dir = tmp_path / "experiment-foo"
    exp_dir.mkdir()
    (exp_dir / "benchmarks").mkdir()
    assert load_metrics(exp_dir) is None


def test_load_metrics_returns_record_when_present(tmp_path):
    exp_dir = tmp_path / "experiment-foo"
    bench_dir = exp_dir / "benchmarks"
    bench_dir.mkdir(parents=True)
    payload = {"metrics": {"delta": 0.5}, "metadata": {"tool": "test"}}
    with open(bench_dir / METRICS_FILENAME, "w") as f:
        yaml.safe_dump(payload, f)

    rec = load_metrics(exp_dir)
    assert rec is not None
    assert rec.metrics == {"delta": 0.5}
    assert rec.metadata == {"tool": "test"}


def test_load_metrics_when_no_benchmarks_dir(tmp_path):
    """Even the benchmarks dir is allowed to be absent."""
    exp_dir = tmp_path / "experiment-foo"
    exp_dir.mkdir()
    assert load_metrics(exp_dir) is None
