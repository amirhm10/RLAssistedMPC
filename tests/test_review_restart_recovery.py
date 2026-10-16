"""Regression checks for app-independent startup and interrupted-data preservation."""
from pathlib import Path
from unittest.mock import patch

import pytest

import review_restart_recovery as recovery


@pytest.mark.parametrize("inside_job", [False, True])
def test_interactive_scheduler_ownership_allows_its_own_job(inside_job):
    assert recovery.isolated_interactive_context({"parent_is_scheduler_service": True,
                                                  "session_id": 2, "in_windows_job": inside_job})


@pytest.mark.parametrize("context", [
    {"parent_is_scheduler_service": False, "session_id": 2, "in_windows_job": False},
    {"parent_is_scheduler_service": True, "session_id": 0},
    {"session_id": 2},
])
def test_visible_or_job_free_process_alone_is_insufficient(context):
    assert not recovery.isolated_interactive_context(context)


def test_prepare_rejects_outside_destination_before_inspection(tmp_path):
    target = tmp_path / "unrelated"
    with patch.object(recovery, "ROOT", tmp_path / "repo"), patch.object(recovery, "preflight") as preflight:
        with pytest.raises(ValueError, match="dedicated workspace"):
            recovery.prepare(target)
    preflight.assert_not_called()
    assert not target.exists()


def test_interruption_archive_copies_without_removing_raw_data(tmp_path):
    polymer, column, e3 = (tmp_path / name for name in ("polymer", "column", "e3"))
    output = tmp_path / "recovery"
    for path in (polymer, column, e3, output):
        path.mkdir()
    for stage in recovery.STAGES:
        path = polymer / stage
        path.mkdir()
        (path / "worker.log").write_text("last completed cycle\n")
        checkpoint = path / "training_checkpoints"
        checkpoint.mkdir()
        (checkpoint / "state.pkl").write_bytes(b"preserved full state")
    for name in ("manifest.json", "visible_launches.json"):
        (column / name).write_text("{}")
    for method in ("sg", "threshold", "ls", "bo", "e1_pilot"):
        path = column / method
        path.mkdir()
        (path / "execution.json").write_text('{"state":"running"}')
        (path / "a1").mkdir()
        (path / "a1/raw.npz").write_bytes(b"original interrupted data")
    (e3 / "manifest.json").write_text("{}")
    with patch.multiple(recovery, POLYMER=polymer, COLUMN=column, E3=e3):
        recovery.archive_before_restart(output)
    saved = output / "preserved_before_restart"
    assert (saved / "polymer/e1_combined/worker.log").read_text() == "last completed cycle\n"
    assert (polymer / "e1_combined/training_checkpoints/state.pkl").read_bytes() == b"preserved full state"
    assert (column / "sg/a1/raw.npz").read_bytes() == b"original interrupted data"
    assert (polymer / "e2_sg/worker.log").exists()
