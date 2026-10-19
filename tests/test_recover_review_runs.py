"""Recovery keeps validated work, rejects duplicates and isolates runtime dispatch."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import recover_review_runs as recovery
import run_distillation_phase2_repeats as repeat


def fixture_manifest(tmp_path):
    jobs = []
    for category, state in [("horizon", "completed"), ("markov", "failed"), ("residual", "pending")]:
        directory = tmp_path / category / "a1"
        directory.mkdir(parents=True)
        result = directory / "input_data.pkl"
        job = {"category": category, "mode": "sg", "seed": 7, "state": state,
               "attempts": [{"directory": str(directory), "pid": 123}] if state != "pending" else []}
        if state == "completed":
            result.write_bytes(b"saved scientific output")
            status = {"state": "completed", "result_file": str(result), "aspen_closed": True, "episodes": 300}
            recovery.write(directory / "worker_status.json", status)
            job["result_file"] = str(result)
        jobs.append(job)
    return {"state": "failed", "jobs": jobs, "source_hashes": {"original.py": "hash"}}


def test_recovery_reuses_completed_result_and_preserves_original_manifest(tmp_path):
    old = fixture_manifest(tmp_path)
    original = deepcopy(old)
    with patch.object(recovery, "ensure_stopped"), patch.object(repeat, "source_hashes", return_value=old["source_hashes"]):
        result = recovery.repeat_manifest(old, tmp_path)
    assert old == original
    assert [j["state"] for j in result["jobs"]] == ["completed", "pending", "pending"]
    assert result["jobs"][1]["attempts"] == old["jobs"][1]["attempts"]
    assert len(result["reused_completed_results"]) == 1
    assert result["reused_completed_results"][0]["sha256"]


def test_recovery_rejects_incomplete_previously_completed_run(tmp_path):
    old = fixture_manifest(tmp_path)
    Path(old["jobs"][0]["result_file"]).unlink()
    with patch.object(recovery, "ensure_stopped"), patch.object(repeat, "source_hashes", return_value=old["source_hashes"]):
        with pytest.raises(RuntimeError, match="no longer validates"):
            recovery.repeat_manifest(old, tmp_path)


def test_recovery_rejects_changed_scientific_sources(tmp_path):
    old = fixture_manifest(tmp_path)
    with patch.object(repeat, "source_hashes", return_value={"original.py": "changed"}):
        with pytest.raises(RuntimeError, match="sources changed"):
            recovery.repeat_manifest(old, tmp_path)


def test_live_worker_prevents_recovery(tmp_path):
    old = fixture_manifest(tmp_path)
    with patch.object(recovery, "ensure_stopped", side_effect=RuntimeError("Live process")), \
         patch.object(repeat, "source_hashes", return_value=old["source_hashes"]):
        with pytest.raises(RuntimeError, match="Live process"):
            recovery.repeat_manifest(old, tmp_path)


def test_worker_records_actual_conservative_dispatch_before_scientific_run(tmp_path):
    calls = []
    from contextlib import nullcontext
    with patch("run_polymer_e3_recovery.environment", return_value={"TASK_TEST_DISPATCH": "set"}), \
         patch("run_polymer_e3_recovery.runtime_info", return_value={"verified": True}), \
         patch("column_review.common.record_owned_apps", return_value=nullcontext()), \
         patch.object(repeat, "run_worker", side_effect=lambda p: calls.append(p) or 0):
        assert recovery.repeat_worker(tmp_path / "job.json") == 0
    assert json.loads((tmp_path / "isolated_runtime.json").read_text()) == {"verified": True}
    assert calls == [tmp_path / "job.json"]


def test_progress_prefers_recovery_and_does_not_show_old_failure_as_current(tmp_path):
    from revision_comparisons.progress import snapshot, display
    from contextlib import redirect_stdout
    import io
    current = tmp_path / "current"
    recovery.write(tmp_path / "Distillation/Results/review_recovery/latest.json",
                   {"repeats": str(current), "column": str(tmp_path / "new_column")})
    recovery.write(current / "manifest.json", {"jobs": [{"category": "markov", "state": "pending"}]})
    data = snapshot(tmp_path)
    assert data["counts"]["column_repeats"] == {"pending": 1}
    data["historical_studies"] = ["old_failed"]
    data["counts"]["old_failed"] = {"failed": 5}
    text = io.StringIO()
    with redirect_stdout(text):
        display(data)
    assert "old_failed" not in text.getvalue()
    assert "column_repeats" in text.getvalue()
