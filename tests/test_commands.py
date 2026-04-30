"""End-to-end tests for `castra exp create / fork / repair / list / info`."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from castra import paths
from castra.cli import main
from castra.spec import ExperimentSpec


def _run(args: list[str]) -> tuple[int, str]:
    """Run the click CLI inside the current cwd; return (exit_code, output)."""
    runner = CliRunner()
    result = runner.invoke(main, args, catch_exceptions=False)
    return result.exit_code, result.output


def test_create_makes_worktree_and_config(project_repo: Path) -> None:
    code, out = _run(["exp", "create", "exp-1"])
    assert code == 0, out
    wt = project_repo.parent / "myproj-exp-1"
    assert wt.exists()
    config = wt / "experiments" / "exp-1" / "config.yaml"
    assert config.exists()
    spec = ExperimentSpec.from_yaml(config)
    assert spec.name == "exp-1"
    assert spec.parent is None
    assert spec.created_at is not None
    assert spec.git_sha is not None


def test_create_with_overrides(project_repo: Path) -> None:
    code, out = _run([
        "exp", "create", "exp-1",
        "-o", "axes.nn.hidden_dim=128",
        "-o", "env.num_players=4",
    ])
    assert code == 0, out
    config = project_repo.parent / "myproj-exp-1" / "experiments" / "exp-1" / "config.yaml"
    spec = ExperimentSpec.from_yaml(config)
    assert spec.axes == {"nn": {"hidden_dim": 128}}
    assert spec.env == {"num_players": 4}


def test_create_scaffolds_subdirs(project_repo: Path) -> None:
    code, _ = _run(["exp", "create", "exp-1"])
    assert code == 0
    base = project_repo.parent / "myproj-exp-1" / "experiments" / "exp-1"
    for sub in ("checkpoints", "runs", "benchmarks", "logs"):
        assert (base / sub).is_dir(), f"missing {sub}"


def test_create_refuses_duplicate(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    code, out = _run(["exp", "create", "exp-1"])
    assert code != 0
    assert "already exists" in out


def test_list_shows_created(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    _run(["exp", "create", "exp-2"])
    code, out = _run(["exp", "list"])
    assert code == 0
    assert "exp-1" in out
    assert "exp-2" in out


def test_list_empty(project_repo: Path) -> None:
    code, out = _run(["exp", "list"])
    assert code == 0
    assert "no experiments yet" in out


def test_list_shows_archived_experiments(project_repo: Path) -> None:
    """After create + finalize + ship + archive, the experiment still
    appears in `exp list` with status=archived."""
    _run(["exp", "create", "exp-1"])
    _run(["exp", "finalize", "exp-1", "--gen", "5", "--reason", "done"])
    _run(["exp", "ship", "exp-1"])
    _run(["exp", "archive", "exp-1"])
    code, out = _run(["exp", "list"])
    assert code == 0, out
    assert "exp-1" in out
    assert "archived" in out


def test_list_shows_shipped_only_experiments(project_repo: Path) -> None:
    """Experiments written directly to <project_root>/experiments/<name>/
    (no worktree, no branch) show up with status=shipped."""
    # Write a config.yaml directly into the project root's experiments dir.
    shipped = project_repo / "experiments" / "direct-write"
    shipped.mkdir(parents=True)
    (shipped / "config.yaml").write_text(
        "name: direct-write\n"
        "parent: null\n"
        "parent_checkpoint: null\n"
        "axes: {}\n"
        "env: {}\n"
        "shards: {}\n"
        "artifacts: {}\n"
        "concluded_gen: null\n"
        "concluded_reason: null\n"
        "concluded_at: null\n"
        "created_at: null\n"
        "git_sha: null\n"
    )

    code, out = _run(["exp", "list"])
    assert code == 0, out
    assert "direct-write" in out
    assert "shipped" in out


def test_list_distinguishes_active_archived_shipped(project_repo: Path) -> None:
    """All three statuses can coexist; verify each row gets the right one."""
    # Active.
    _run(["exp", "create", "active-one"])

    # Archived (create -> finalize -> ship -> archive).
    _run(["exp", "create", "archived-one"])
    _run(["exp", "finalize", "archived-one", "--gen", "1", "--reason", "x"])
    _run(["exp", "ship", "archived-one"])
    _run(["exp", "archive", "archived-one"])

    # Shipped-only (direct write).
    shipped = project_repo / "experiments" / "shipped-only"
    shipped.mkdir(parents=True)
    (shipped / "config.yaml").write_text("name: shipped-only\n")

    code, out = _run(["exp", "list"])
    assert code == 0, out

    lines = out.splitlines()
    def _line_for(name):
        for line in lines:
            if name in line and "STATUS" not in line:
                return line
        raise AssertionError(f"{name!r} not found in:\n{out}")

    assert "active" in _line_for("active-one")
    assert "archived" in _line_for("archived-one")
    assert "shipped" in _line_for("shipped-only")


def test_info_shows_spec(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1", "-o", "env.num_players=4"])
    code, out = _run(["exp", "info", "exp-1"])
    assert code == 0
    assert "exp-1" in out
    assert "experiment/exp-1" in out


def test_info_unknown_fails(project_repo: Path) -> None:
    code, out = _run(["exp", "info", "ghost"])
    assert code != 0
    assert "no experiment" in out.lower()


def test_fork_inherits_axes(project_repo: Path) -> None:
    _run(["exp", "create", "parent", "-o", "axes.nn.hidden_dim=128"])
    code, out = _run(["exp", "fork", "child", "--from", "parent"])
    assert code == 0, out
    child_config = (
        project_repo.parent / "myproj-child" / "experiments" / "child" / "config.yaml"
    )
    spec = ExperimentSpec.from_yaml(child_config)
    assert spec.parent == "parent"
    assert spec.parent_checkpoint == "auto"
    assert spec.axes == {"nn": {"hidden_dim": 128}}


def test_fork_branches_off_parent(project_repo: Path) -> None:
    """When parent's experiment branch exists, child should branch off it,
    NOT off main — so parent's code changes are inherited.
    """
    import subprocess

    _run(["exp", "create", "parent"])
    parent_wt = project_repo.parent / "myproj-parent"
    # Make a unique commit on parent's branch.
    (parent_wt / "PARENT_MARKER.txt").write_text("hello")
    subprocess.run(["git", "add", "-A"], cwd=str(parent_wt), check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t.t", "-c", "user.name=t",
         "commit", "-m", "parent change"],
        cwd=str(parent_wt), check=True,
    )

    _run(["exp", "fork", "child", "--from", "parent"])
    child_wt = project_repo.parent / "myproj-child"
    # The marker file should be present in child since it forked from parent's branch.
    assert (child_wt / "PARENT_MARKER.txt").exists()


def test_fork_falls_back_to_main_when_parent_branch_missing(project_repo: Path) -> None:
    """If we never `created` the source, fork should still succeed off main."""
    code, out = _run(["exp", "fork", "child", "--from", "ghost"])
    assert code == 0, out
    config = project_repo.parent / "myproj-child" / "experiments" / "child" / "config.yaml"
    spec = ExperimentSpec.from_yaml(config)
    assert spec.parent == "ghost"


def test_fork_inherits_parent_checkpoint_file(project_repo: Path) -> None:
    """If parent has experiments/<name>/checkpoints/latest.pt, fork copies it."""
    _run(["exp", "create", "parent"])
    parent_ckpts = (
        project_repo.parent / "myproj-parent" / "experiments" / "parent" / "checkpoints"
    )
    parent_ckpts.mkdir(parents=True, exist_ok=True)
    (parent_ckpts / "latest.pt").write_bytes(b"fake-checkpoint-bytes")

    _run(["exp", "fork", "child", "--from", "parent"])
    child_ckpt = (
        project_repo.parent / "myproj-child" / "experiments" / "child"
        / "checkpoints" / "latest.pt"
    )
    assert child_ckpt.exists()
    assert child_ckpt.read_bytes() == b"fake-checkpoint-bytes"


def test_repair_succeeds_after_create(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    code, out = _run(["exp", "repair", "exp-1", "--no-rebuild"])
    assert code == 0, out


def test_repair_unknown_fails(project_repo: Path) -> None:
    code, out = _run(["exp", "repair", "ghost", "--no-rebuild"])
    assert code != 0


# --- Phase 8: finalize / ship / archive / promote / compare -----------------


def test_finalize_writes_concluded_metadata(project_repo: Path) -> None:
    code, _ = _run(["exp", "create", "exp-1"])
    assert code == 0
    code, out = _run([
        "exp", "finalize", "exp-1",
        "--gen", "42", "--reason", "converged at val 0.019",
    ])
    assert code == 0, out
    config = project_repo.parent / "myproj-exp-1" / "experiments" / "exp-1" / "config.yaml"
    spec = ExperimentSpec.from_yaml(config)
    assert spec.concluded_gen == 42
    assert spec.concluded_reason == "converged at val 0.019"
    assert spec.concluded_at is not None


def test_finalize_refuses_overwrite_without_force(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    _run(["exp", "finalize", "exp-1", "--gen", "10", "--reason", "first"])
    code, out = _run([
        "exp", "finalize", "exp-1", "--gen", "20", "--reason", "again"
    ])
    assert code != 0
    assert "already finalized" in out


def test_finalize_force_overwrites(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    _run(["exp", "finalize", "exp-1", "--gen", "10", "--reason", "first"])
    code, _ = _run([
        "exp", "finalize", "exp-1",
        "--gen", "20", "--reason", "redo", "--force",
    ])
    assert code == 0
    config = project_repo.parent / "myproj-exp-1" / "experiments" / "exp-1" / "config.yaml"
    spec = ExperimentSpec.from_yaml(config)
    assert spec.concluded_gen == 20
    assert spec.concluded_reason == "redo"


def test_finalize_unknown_experiment(project_repo: Path) -> None:
    code, out = _run(["exp", "finalize", "ghost", "--gen", "1", "--reason", "x"])
    assert code != 0
    assert "no experiment named" in out


def test_ship_copies_config_and_benchmarks(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    # Drop a fake benchmark file inside the worktree so ship has something to copy.
    bench_dir = (project_repo.parent / "myproj-exp-1"
                 / "experiments" / "exp-1" / "benchmarks")
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "tournament.jsonl").write_text("{\"score\": 1.0}\n")
    _run(["exp", "finalize", "exp-1", "--gen", "5", "--reason", "done"])
    code, out = _run(["exp", "ship", "exp-1"])
    assert code == 0, out
    shipped_config = project_repo / "experiments" / "exp-1" / "config.yaml"
    shipped_bench = project_repo / "experiments" / "exp-1" / "benchmarks" / "tournament.jsonl"
    assert shipped_config.exists()
    assert shipped_bench.exists()
    spec = ExperimentSpec.from_yaml(shipped_config)
    assert spec.concluded_gen == 5


def test_ship_refuses_without_finalize(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    code, out = _run(["exp", "ship", "exp-1"])
    assert code != 0
    assert "not finalized" in out


def test_ship_force_skips_finalize_check(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    code, _ = _run(["exp", "ship", "exp-1", "--force"])
    assert code == 0


def test_archive_removes_worktree(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    _run(["exp", "finalize", "exp-1", "--gen", "5", "--reason", "done"])
    wt = project_repo.parent / "myproj-exp-1"
    assert wt.exists()
    code, out = _run(["exp", "archive", "exp-1"])
    assert code == 0, out
    assert not wt.exists()


def test_archive_refuses_without_finalize(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    code, out = _run(["exp", "archive", "exp-1"])
    assert code != 0
    assert "not finalized" in out


def test_archive_force_skips_check(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    wt = project_repo.parent / "myproj-exp-1"
    assert wt.exists()
    code, _ = _run(["exp", "archive", "exp-1", "--force"])
    assert code == 0
    assert not wt.exists()


def test_archive_idempotent_on_already_archived(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    _run(["exp", "archive", "exp-1", "--force"])
    code, out = _run(["exp", "archive", "exp-1"])
    # already archived -> no-op success
    assert code == 0
    assert "already archived" in out


def test_promote_copies_checkpoint_and_updates_frontier_md(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    ckpt = (project_repo.parent / "myproj-exp-1"
            / "experiments" / "exp-1" / "checkpoints" / "gen-7.pt")
    ckpt.write_bytes(b"fake-checkpoint-bytes")
    _run(["exp", "finalize", "exp-1", "--gen", "7", "--reason", "best val"])
    code, out = _run(["exp", "promote", "exp-1"])
    assert code == 0, out
    promoted = project_repo / "frontier" / "exp-1-gen7.pt"
    assert promoted.exists()
    assert promoted.read_bytes() == b"fake-checkpoint-bytes"
    frontier_md = project_repo / "FRONTIER.md"
    assert frontier_md.exists()
    text = frontier_md.read_text()
    assert "exp-1" in text
    assert "gen=7" in text
    assert "best val" in text


def test_promote_falls_back_to_latest_pt(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    ckpt = (project_repo.parent / "myproj-exp-1"
            / "experiments" / "exp-1" / "checkpoints" / "latest.pt")
    ckpt.write_bytes(b"latest-bytes")
    _run(["exp", "finalize", "exp-1", "--gen", "9", "--reason", "x"])
    code, out = _run(["exp", "promote", "exp-1"])
    assert code == 0, out
    assert "falling back to latest.pt" in out
    promoted = project_repo / "frontier" / "exp-1-gen9.pt"
    assert promoted.exists()
    assert promoted.read_bytes() == b"latest-bytes"


def test_promote_explicit_gen_override(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    ck_dir = (project_repo.parent / "myproj-exp-1"
              / "experiments" / "exp-1" / "checkpoints")
    (ck_dir / "gen-3.pt").write_bytes(b"gen-3")
    (ck_dir / "gen-7.pt").write_bytes(b"gen-7")
    _run(["exp", "finalize", "exp-1", "--gen", "7", "--reason", "x"])
    code, _ = _run(["exp", "promote", "exp-1", "--gen", "3"])
    assert code == 0
    assert (project_repo / "frontier" / "exp-1-gen3.pt").exists()


def test_promote_refuses_without_finalize(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    ck_dir = (project_repo.parent / "myproj-exp-1"
              / "experiments" / "exp-1" / "checkpoints")
    (ck_dir / "latest.pt").write_bytes(b"x")
    code, out = _run(["exp", "promote", "exp-1"])
    assert code != 0
    assert "not finalized" in out


def test_compare_emits_table(project_repo: Path) -> None:
    _run(["exp", "create", "exp-a"])
    _run(["exp", "create", "exp-b"])
    _run(["exp", "finalize", "exp-a", "--gen", "5", "--reason", "alpha done"])
    _run(["exp", "finalize", "exp-b", "--gen", "8", "--reason", "beta done"])
    code, out = _run(["exp", "compare", "exp-a", "exp-b"])
    assert code == 0, out
    assert "exp-a" in out and "exp-b" in out
    assert "alpha done" in out
    assert "beta done" in out


def _write_metrics(worktree_path: Path, name: str,
                   metrics: dict, metadata: dict | None = None) -> None:
    from castra.metrics import MetricsRecord
    bench = worktree_path / "experiments" / name / "benchmarks"
    bench.mkdir(parents=True, exist_ok=True)
    rec = MetricsRecord(metrics=metrics, metadata=metadata or {})
    rec.to_yaml(bench / "metrics.yaml")


def test_compare_tabulates_metrics_with_delta(project_repo: Path) -> None:
    """With exactly two experiments, compare shows a metrics section with
    a Δ column."""
    _run(["exp", "create", "exp-a"])
    _run(["exp", "create", "exp-b"])
    wt_a = project_repo.parent / "myproj-exp-a"
    wt_b = project_repo.parent / "myproj-exp-b"
    _write_metrics(wt_a, "exp-a",
                   metrics={"delta": -0.9, "wins": 24},
                   metadata={"search": "greedy"})
    _write_metrics(wt_b, "exp-b",
                   metrics={"delta": 3.71, "wins": 79},
                   metadata={"search": "puct"})

    code, out = _run(["exp", "compare", "exp-a", "exp-b"])
    assert code == 0, out
    assert "=== metrics ===" in out
    assert "delta" in out
    assert "wins" in out
    # Δ column appears for two-experiment compare.
    assert " diff " in out  # diff column header (the Δ-equivalent)
    # Metadata section also rendered.
    assert "=== metadata ===" in out
    assert "search" in out
    assert "puct" in out and "greedy" in out


def test_compare_three_experiments_omits_delta_column(project_repo: Path) -> None:
    """3+ experiments: no Δ column (delta only well-defined pairwise)."""
    for n in ("exp-a", "exp-b", "exp-c"):
        _run(["exp", "create", n])
        wt = project_repo.parent / f"myproj-{n}"
        _write_metrics(wt, n, metrics={"score": 1.0})

    code, out = _run(["exp", "compare", "exp-a", "exp-b", "exp-c"])
    assert code == 0, out
    assert "=== metrics ===" in out
    # Headers contain all three names but no Δ.
    metrics_section = out.split("=== metrics ===")[1]
    header_line = metrics_section.split("\n", 2)[1]
    # No diff / Δ column when 3+ experiments are compared.
    cols = header_line.split()
    assert "diff" not in cols


def test_compare_handles_missing_metrics_with_dash(project_repo: Path) -> None:
    """An experiment without metrics.yaml shows '—' in the metrics rows."""
    _run(["exp", "create", "exp-a"])
    _run(["exp", "create", "exp-b"])
    wt_a = project_repo.parent / "myproj-exp-a"
    _write_metrics(wt_a, "exp-a", metrics={"score": 1.5})
    # exp-b deliberately has no metrics file.

    code, out = _run(["exp", "compare", "exp-a", "exp-b"])
    assert code == 0, out
    assert "=== metrics ===" in out
    # Missing values show as "-".
    assert " -  " in out or "  - " in out


def test_compare_omits_metrics_section_when_none_present(project_repo: Path) -> None:
    """No metrics anywhere: compare prints only the spec table (existing
    behavior preserved)."""
    _run(["exp", "create", "exp-a"])
    _run(["exp", "create", "exp-b"])
    code, out = _run(["exp", "compare", "exp-a", "exp-b"])
    assert code == 0, out
    assert "=== metrics ===" not in out
    assert "=== metadata ===" not in out


def test_compare_requires_two_names(project_repo: Path) -> None:
    _run(["exp", "create", "exp-a"])
    code, out = _run(["exp", "compare", "exp-a"])
    assert code != 0
    assert "needs at least two" in out


def test_info_reflects_finalization(project_repo: Path) -> None:
    _run(["exp", "create", "exp-1"])
    _run(["exp", "finalize", "exp-1", "--gen", "12", "--reason", "complete"])
    code, out = _run(["exp", "info", "exp-1"])
    assert code == 0
    assert "concluded:" in out
    assert "gen=12" in out
    assert "complete" in out
    code, out = _run(["exp", "list"])
    # status column should show 'final'
    assert "final" in out
