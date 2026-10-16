"""Isolation and preservation contracts for the failed E3 combined restart."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

import run_polymer_e3_recovery as recovery
import run_e3_prediction_quality as e3


def fixture_manifest():
    return {"options": {"plants": ["polymer"], "smoke": False, "seed": 7},
            "python": "same-python", "source_hashes": {"controller": "unchanged"},
            "jobs": [{"category": category, "id": "polymer_"+category, "state": state,
                      "attempts": [{"pid": 123, "directory": "old_attempt"}]}
                     for category, state in (("nominal", "completed"), ("markov", "completed"), ("combined", "failed"))]}


def setup_plan(tmp_path, monkeypatch, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(e3, "completed_job", lambda job: job["state"] == "completed")
    monkeypatch.setattr(e3, "pid_alive", lambda pid: False)
    monkeypatch.setattr(e3, "check_sources", lambda expected: None)


def test_environment_override_is_confined_to_a_copy():
    base = {"PATH": "keep", "OPENBLAS_CORETYPE": "HASWELL", "OTHER": "keep"}
    before = deepcopy(base)
    process_before = dict(os.environ)
    selected = recovery.environment(base)
    assert selected["OPENBLAS_CORETYPE"] == "NEHALEM" and selected["PATH"] == "keep"
    assert base == before and dict(os.environ) == process_before


def test_plan_skips_completed_jobs_and_preserves_manifest_and_options(tmp_path, monkeypatch):
    manifest = fixture_manifest()
    setup_plan(tmp_path, monkeypatch, manifest)
    before = (tmp_path / "manifest.json").read_bytes()
    actual, plan = recovery.plan(tmp_path)
    assert actual == manifest
    assert plan["restart_only"] == "polymer_combined"
    assert plan["skip_verified"] == ["polymer_nominal", "polymer_markov"]
    assert plan["options_unchanged"] == manifest["options"]
    assert (tmp_path / "manifest.json").read_bytes() == before
    assert list(tmp_path.iterdir()) == [tmp_path / "manifest.json"]


@pytest.mark.parametrize("problem", ["live", "distillation", "completed_combined", "invalid_completed", "source_change"])
def test_recovery_rejects_wrong_scope_duplicates_and_invalid_provenance(tmp_path, monkeypatch, problem):
    manifest = fixture_manifest()
    if problem == "distillation": manifest["options"]["plants"] = ["distillation"]
    if problem == "completed_combined": manifest["jobs"][-1]["state"] = "completed"
    setup_plan(tmp_path, monkeypatch, manifest)
    if problem == "live": monkeypatch.setattr(e3, "pid_alive", lambda pid: True)
    if problem == "invalid_completed": monkeypatch.setattr(e3, "completed_job", lambda job: False)
    if problem == "source_change":
        def reject(expected): raise RuntimeError("source changed")
        monkeypatch.setattr(e3, "check_sources", reject)
    with pytest.raises((ValueError, RuntimeError)):
        recovery.plan(tmp_path)
