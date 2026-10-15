"""Regression contracts for export/source-guard failures; no production runs."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

import run_polymer_e1 as e1
import run_polymer_comparisons as e2
import run_polymer_review_recovery as recovery
from comparison_studies.checkpoints import load_inference_checkpoint, save_inference_checkpoint


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("scope", ["e1", "e2"])
def test_unrelated_experiment_changes_do_not_invalidate_guard_but_dependencies_do(tmp_path, monkeypatch, scope):
    import comparison_studies.common as common
    monkeypatch.setattr(common, "ROOT", tmp_path)
    _write(tmp_path / "run_polymer_e1.py", "# wrapper")
    _write(tmp_path / "run_polymer_comparisons.py", "# comparison wrapper")
    _write(tmp_path / "review_experiments/__init__.py", "# opt-in")
    active = tmp_path / "review_experiments/e1/runtime.py"
    _write(active, "# original runtime")
    _write(tmp_path / "comparison_studies/native.py", "# native comparator")
    hashes = (lambda: e1.source_hashes(tmp_path)) if scope == "e1" else common.source_hashes
    before = hashes()
    unrelated = tmp_path / "review_experiments/e3/runtime.py"
    _write(unrelated, "# new E3 experiment")
    assert hashes() == before
    _write(unrelated, "# edited E3 experiment")
    assert hashes() == before
    _write(active, "# CHANGED runtime")
    assert hashes() != before
    _write(active, "# original runtime")
    _write(tmp_path / "review_experiments/e1/new_dependency.py", "# newly added E1 dependency")
    assert hashes() != before


def _real_agents():
    from DQN.supervisor_gated_dqn_agent import SupervisorGatedDQNAgent
    from TD3Agent.supervisor_gated_agent import SupervisorGatedTD3Agent
    dqn = SupervisorGatedDQNAgent(3, 4, [8, 8], batch_size=4, buffer_size=32, device=torch.device("cpu"))
    td3 = SupervisorGatedTD3Agent(3, 2, [8, 8], [8, 8], batch_size=4, buffer_size=32, device=torch.device("cpu"))
    return {"horizon_agent": dqn, "weight_agent": td3}


def test_registered_dqn_td3_weights_roundtrip_preserves_predictions_rng_and_live_state(tmp_path):
    agents = _real_agents()
    inputs = torch.tensor([[.1, -.2, .3]])
    with torch.no_grad():
        expected_q = agents["horizon_agent"].online(inputs).clone()
        expected_action = agents["weight_agent"].actor(inputs).clone()
        expected_critics = tuple(t.clone() for t in agents["weight_agent"].critic(inputs, expected_action))
    before_rng = torch.random.get_rng_state().clone()
    modes = {name: agent.training for name, agent in agents.items()}
    path = tmp_path / "networks.pt"
    report = save_inference_checkpoint(path, agents, metadata={"seed": 7, "smoke": True})
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert {name: agent.training for name, agent in agents.items()} == modes
    assert report["round_trip_verified"] and all(report["tensor_counts"].values())
    payload = load_inference_checkpoint(path)
    assert any(key.startswith("online.") for key in payload["agents"]["horizon_agent"]["state_dict"])
    assert any(key.startswith("critic.") for key in payload["agents"]["weight_agent"]["state_dict"])
    fresh = _real_agents()
    load_inference_checkpoint(path, fresh)
    with torch.no_grad():
        torch.testing.assert_close(fresh["horizon_agent"].online(inputs), expected_q, rtol=0, atol=0)
        torch.testing.assert_close(fresh["weight_agent"].actor(inputs), expected_action, rtol=0, atol=0)
        for actual, expected in zip(fresh["weight_agent"].critic(inputs, expected_action), expected_critics):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        save_inference_checkpoint(path, fresh)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash


@pytest.mark.parametrize("kind", ["empty_agent", "no_agents", "nonfinite"])
def test_invalid_checkpoint_is_rejected_without_leaving_a_final_file(tmp_path, kind):
    agent = torch.nn.Module() if kind == "empty_agent" else torch.nn.Linear(2, 2)
    if kind == "nonfinite":
        with torch.no_grad():
            agent.weight[0, 0] = float("nan")
    path = tmp_path / "invalid.pt"
    with pytest.raises(ValueError):
        save_inference_checkpoint(path, {} if kind == "no_agents" else {"agent": agent})
    assert not path.exists() and not path.with_suffix(".pt.tmp").exists()


def test_legacy_empty_checkpoint_cannot_be_misreported_as_valid(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"horizon_agent": {}, "weight_agent": {}}, path)
    with pytest.raises(ValueError, match="legacy empty"):
        load_inference_checkpoint(path)


@pytest.mark.parametrize("study", ["e1", "e2"])
def test_completed_trajectory_survives_final_source_check_failure(tmp_path, monkeypatch, study):
    import comparison_studies.common as common
    import comparison_studies.native as native
    import review_experiments.e1.runtime as runtime
    monkeypatch.chdir(tmp_path)
    output = tmp_path / study
    output.mkdir()
    expected = {"dependency.py": "before"}
    calls = iter([expected, {"dependency.py": "after"}])
    job = {"source_hashes": expected, "seed": 7, "threads": 1, "smoke": True}
    marker = {"nFE": 800, "test_raw_signal": np.arange(4)}
    if study == "e1":
        job.update(category="combined", options={"horizons": [4, 8]})
        monkeypatch.setattr(e1, "source_hashes", lambda: next(calls))
        monkeypatch.setattr(e1, "execute_entrypoint", lambda *a: {"TRAINING_PROFILE_NAME": "robustness_200_100", "result_bundle": marker})
        monkeypatch.setattr(runtime, "install_runner", lambda *a: nullcontext())
        function = e1.worker
    else:
        job.update(method="ls", profile="legacy_gradual_200")
        monkeypatch.setattr(common, "source_hashes", lambda: next(calls))
        monkeypatch.setattr(common, "prepare_nominal", lambda *a: {"NB": {}})
        monkeypatch.setattr(common, "episode_bundle", lambda *a: {"nFE": 800})
        monkeypatch.setattr(common, "solver_audit", lambda: nullcontext([]))
        monkeypatch.setattr(native, "ls_run", lambda *a: marker)
        function = e2.worker
    path = output / "job.json"
    path.write_text(json.dumps(job))
    assert function(path) == 1
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "failed" and status["trajectory_saved"]
    assert status["trajectory_sha256"] == hashlib.sha256((output / "training.pkl").read_bytes()).hexdigest()
    with (output / "training.pkl").open("rb") as stream:
        result = pickle.load(stream)
    np.testing.assert_array_equal(result["test_raw_signal"], np.arange(4))


def test_recovery_source_audit_allows_documented_wrapper_fix_but_rejects_scientific_change(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    wrapper = tmp_path / "run_polymer_e1.py"
    dependency = tmp_path / "utils/combined_runner.py"
    _write(wrapper, "old wrapper")
    _write(dependency, "original controller")
    expected = {"run_polymer_e1.py": recovery.sha256(wrapper), "utils/combined_runner.py": recovery.sha256(dependency)}
    _write(wrapper, "fixed wrapper")
    report = recovery.audit_historical_sources(expected)
    assert set(report["documented_wrapper_changes"]) == {"run_polymer_e1.py"}
    _write(dependency, "modified controller")
    with pytest.raises(ValueError, match="combined_runner"):
        recovery.audit_historical_sources(expected)


def test_recovery_dry_run_never_starts_a_process_or_creates_outputs(tmp_path, monkeypatch, capsys):
    plan = {"stages": [{"name": "e1_combined", "command": ["e1"]}, {"name": "e2_sg", "command": ["e2"]}]}
    monkeypatch.setattr(recovery, "build_plan", lambda *a, **k: plan)
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *a, **k: pytest.fail("Dry run started a process"))
    output = tmp_path / "unused"
    assert recovery.main(["--dry-run", "--stages", "e2_sg", "--output-dir", str(output)]) == 0
    assert not output.exists()
    assert '"name": "e1_combined"' not in capsys.readouterr().out


def test_recovery_failure_stops_before_starting_the_next_stage(tmp_path, monkeypatch):
    plan = {"stages": [{"name": "e1_combined", "command": ["e1"]}, {"name": "e2_sg", "command": ["e2"]}]}
    monkeypatch.setattr(recovery, "build_plan", lambda *a, **k: plan)
    calls = []
    def fail_first(command, directory, name, offsets):
        calls.append(name)
        return 9
    monkeypatch.setattr(recovery, "run_stage", fail_first)
    output = tmp_path / "new_attempt"
    assert recovery.main(["--run", "--output-dir", str(output)]) == 9
    assert calls == ["e1_combined"]
    assert json.loads((output / "e1_combined_exit.json").read_text())["returncode"] == 9
    assert not (output / "e2_sg_exit.json").exists()


def test_recovery_refuses_to_overwrite_an_existing_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "build_plan", lambda *a, **k: {"stages": []})
    output = tmp_path / "old_attempt"
    output.mkdir()
    sentinel = output / "original.txt"
    sentinel.write_text("preserve me")
    with pytest.raises(FileExistsError):
        recovery.main(["--run", "--output-dir", str(output)])
    assert sentinel.read_text() == "preserve me"
    assert set(output.iterdir()) == {sentinel}


def test_zero_exit_with_incomplete_matched_decisions_is_not_certified(tmp_path):
    attempt = tmp_path / "timestamp/combined_s7/a1"
    _write(attempt / "status.json", json.dumps({"state": "completed", "failed_pairs": 0, "decisions": 16}))
    _write(attempt / "capture/capture_status.json", json.dumps({"state": "completed", "saved_decisions": 15}))
    with pytest.raises(ValueError, match="incomplete or failed"):
        recovery.verify_stage_outputs("e1_combined", tmp_path, True)
