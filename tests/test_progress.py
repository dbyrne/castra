"""Progress tracking — atomic write, read, stale detection."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from castra.progress import (
    PROGRESS_FILENAME,
    STALE_AFTER_SECONDS,
    Progress,
    clear_progress,
    read_progress,
    update_progress,
)


def test_round_trip(tmp_path):
    update_progress(tmp_path, stage="self-play", message="32/100 shards")
    p = read_progress(tmp_path)
    assert p is not None
    assert p.stage == "self-play"
    assert p.message == "32/100 shards"
    assert p.updated_at  # ISO timestamp present
    assert p.age_seconds is not None and p.age_seconds < 1.0
    assert not p.is_stale


def test_returns_none_when_missing(tmp_path):
    assert read_progress(tmp_path) is None


def test_returns_none_when_no_dir(tmp_path):
    assert read_progress(tmp_path / "doesnt-exist") is None


def test_handles_corrupt_json(tmp_path):
    """Reader is defensive — bad file → None (not exception)."""
    (tmp_path / PROGRESS_FILENAME).write_text("not json {{{")
    assert read_progress(tmp_path) is None


def test_stale_detection(tmp_path):
    """An old timestamp marks the progress as stalled."""
    old_iso = (
        _dt.datetime.now(_dt.timezone.utc)
        - _dt.timedelta(seconds=STALE_AFTER_SECONDS + 60)
    ).isoformat()
    payload = {"stage": "gate", "message": "stuck", "updated_at": old_iso}
    (tmp_path / PROGRESS_FILENAME).write_text(json.dumps(payload))
    p = read_progress(tmp_path)
    assert p is not None
    assert p.is_stale


def test_atomic_overwrite(tmp_path):
    """Successive updates fully replace the previous one."""
    update_progress(tmp_path, stage="a", message="first")
    update_progress(tmp_path, stage="b", message="second")
    p = read_progress(tmp_path)
    assert p.stage == "b"
    assert p.message == "second"


def test_extra_is_persisted(tmp_path):
    update_progress(
        tmp_path, stage="done", extra={"final_delta": 12.3, "n_games": 100},
    )
    p = read_progress(tmp_path)
    assert p.extra == {"final_delta": 12.3, "n_games": 100}


def test_creates_dir_if_missing(tmp_path):
    """update_progress mkdirs the experiment_dir if it doesn't exist yet."""
    target = tmp_path / "deep" / "experiment"
    update_progress(target, stage="init")
    assert (target / PROGRESS_FILENAME).exists()


def test_clear_removes_file(tmp_path):
    update_progress(tmp_path, stage="x")
    assert (tmp_path / PROGRESS_FILENAME).exists()
    clear_progress(tmp_path)
    assert not (tmp_path / PROGRESS_FILENAME).exists()
    # Idempotent — second clear is a no-op.
    clear_progress(tmp_path)


def test_clear_is_safe_when_missing(tmp_path):
    clear_progress(tmp_path)  # no exception


def test_progress_age_when_no_timestamp():
    """A Progress with no updated_at can't compute age — returns None."""
    p = Progress(stage="x")
    assert p.updated_dt is None
    assert p.age_seconds is None
    assert not p.is_stale
