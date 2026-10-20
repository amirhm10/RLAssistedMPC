"""Regression tests for isolated recovery, with no live Aspen connection."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from recover_polymer_threshold import AGENTS, PolicyExportManager
from comparison_studies.checkpoints import load_inference_checkpoint
from restart_stopped_markov import allowed_owned_binary
from repair_aspen_snapshot import ReadyNativeSnapshot, verify_continuation_arrays


def test_export_occurs_only_at_completed_cycles():
    manager = PolicyExportManager("unused", {})
    environment = {"time_in_sub_episodes": 800, "nFE": 160000}
    assert not manager.due(0, environment)
    assert not manager.due(798, environment)
    assert manager.due(799, environment)
    assert not manager.due(1599, environment)
    assert manager.due(7999, environment)
    assert manager.due(159999, environment)
    assert not manager.has_checkpoint  # inference export must never imply resume.


def test_all_four_networks_round_trip_without_mutation_or_rng_change(tmp_path):
    agents = {name: torch.nn.Linear(2, 2) for name in AGENTS}
    initial = {name: {k:v.clone() for k,v in a.state_dict().items()} for name,a in agents.items()}
    rng = torch.get_rng_state().clone()
    manager = PolicyExportManager(tmp_path / "policies", {"smoke": True})
    manager.compiled = SimpleNamespace(step_name="k")
    manager.save(dict(agents, k=799, time_in_sub_episodes=800), None)
    payload = load_inference_checkpoint(tmp_path / "policies/cycle_0001.pt")
    assert set(payload["agents"]) == set(AGENTS)
    assert torch.equal(torch.get_rng_state(), rng)
    for name, agent in agents.items():
        for key, value in agent.state_dict().items():
            assert torch.equal(value, initial[name][key])
            assert torch.equal(value, payload["agents"][name]["state_dict"][key])
    assert payload["metadata"]["completed_steps"] == 800
    assert payload["metadata"]["training_resume"] is False


def test_orphan_cleanup_rejects_other_binaries_and_names():
    assert allowed_owned_binary("conhost.exe", "C:/Windows/System32/conhost.exe")
    assert allowed_owned_binary("sim_server.exe", "C:/Program Files/AspenTech/AMSystem V14.0/Bin/sim_server.exe")
    assert not allowed_owned_binary("conhost.exe", "C:/Users/hamediaa/conhost.exe")
    assert not allowed_owned_binary("python.exe", "C:/Windows/System32/python.exe")


@pytest.mark.parametrize("changed", ["clock", "internal"])
def test_restore_initialization_rejects_advancement_or_changed_state(monkeypatch, changed):
    values = np.zeros(16)
    def step(wait, script):
        values[0 if changed == "clock" else 4] = 1
    plant = SimpleNamespace(sim=SimpleNamespace(Step=step))
    monkeypatch.setattr("column_review.pilot.profile", lambda p: ([], values.copy()))
    restored = ReadyNativeSnapshot.__new__(ReadyNativeSnapshot)
    restored.plant, restored.snapshot, restored.priming_records = plant, SimpleNamespace(restore=lambda: {}), []
    with pytest.raises(RuntimeError, match="advanced time or changed"):
        restored.restore()


def test_restore_completes_only_one_zero_time_initialization(monkeypatch):
    calls, values = [], np.zeros(16)
    plant = SimpleNamespace(sim=SimpleNamespace(Step=lambda *args: calls.append(args)))
    monkeypatch.setattr("column_review.pilot.profile", lambda p: ([], values.copy()))
    restored = ReadyNativeSnapshot.__new__(ReadyNativeSnapshot)
    restored.plant, restored.snapshot, restored.priming_records = plant, SimpleNamespace(restore=lambda: {"observer": 3}), []
    assert restored.restore() == {"observer": 3}
    assert calls == [(True, False)]
    assert len(restored.priming_records) == 1


def test_native_array_comparison_accepts_and_rejects_physical_tolerances():
    def run(offset=0):
        return {"outputs":np.ones((24,2))+offset,"internal":np.ones((24,16)),"controller":{"observer":np.ones(3)}}
    verify_continuation_arrays([run(), run(), run()])
    inside=run()
    inside["outputs"][:,1] += 5e-5
    verify_continuation_arrays([run(),inside])
    with pytest.raises(RuntimeError,match="physical outputs"):
        verify_continuation_arrays([run(),run(1e-3)])
