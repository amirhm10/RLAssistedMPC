"""Contracts for isolated E1 decisions and unchanged training semantics."""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

import run_polymer_e1 as cli
from review_experiments.e1.analysis import summarize
from review_experiments.e1.runtime import CompiledRunner, RUNNER_TARGETS, install_runner, rng_state, restore_rng
from review_experiments.e1.study import StudyOptions, StudySession, branch_metrics
from review_experiments.e1.smoke import compressed_smoke_clock


@pytest.mark.parametrize("category", RUNNER_TARGETS)
def test_current_sources_have_explicit_supported_gate_sites(category):
    module, name = RUNNER_TARGETS[category]
    compiled = CompiledRunner(getattr(importlib.import_module(module), name), category)
    assert len(compiled.sites) == (4 if category == "combined" else 1)
    assert {s.channel for s in compiled.sites.values()} == (
        {"horizon", "weights", "markov", "residual"} if category == "combined" else {category}
    )


@pytest.mark.parametrize("category", cli.ENTRIES)
def test_only_selection_seed_output_change_in_full_configuration(category, tmp_path):
    from systems.polymer import get_polymer_notebook_defaults
    family = "horizon_standard" if category == "horizon" else category
    original = get_polymer_notebook_defaults(family)
    before = deepcopy(original)
    configured = cli.configure_defaults(original, category, 7, tmp_path)
    excluded = {"run_mode", "training_profile_override", "results_dir_override", "result_prefix_override",
                "compare_prefix_override", "agent_kind", "combined_agent_mode", *cli.SEED_KEYS}
    for key in original:
        if key in cli.SEED_KEYS and isinstance(original[key], dict):
            assert repr({k: v for k, v in configured[key].items() if k != "seed"}) == repr({k: v for k, v in original[key].items() if k != "seed"})
        elif key not in excluded:
            assert repr(configured[key]) == repr(original[key])
    assert repr(original) == repr(before)
    assert configured["set_points_len_override"] == original["set_points_len_override"]


def test_rng_roundtrip_covers_numpy_python_and_torch():
    state = rng_state()
    values = (random.random(), np.random.rand(), torch.rand(1).item())
    restore_rng(state)
    assert values == (random.random(), np.random.rand(), torch.rand(1).item())


def _clock_fixture(weight_cfg, runtime_ctx):
    agent = runtime_ctx.get("agent")
    nFE = 3
    def read_clock():
        return i
    for i in range(nFE):
        sg_decision = agent.select_action_with_supervisor(np.zeros(1), supervisor_action=np.zeros(1))
    return read_clock()


def test_serialized_private_helper_uses_restored_advancing_clock():
    import cloudpickle
    from review_experiments.e1.runtime import clone_environment
    class Pause:
        def wants(self, key, environment):
            return True
        def is_active(self, key):
            return False
    compiled = CompiledRunner(_clock_fixture, "weights")
    environment = compiled.environment(Pause(), ({}, {"agent": None}), {})
    generator = environment[_clock_fixture.__name__]()
    assert next(generator)[0] == "before"
    private = clone_environment(environment, compiled)
    payload = cloudpickle.loads(cloudpickle.dumps({name: private[name] for name in compiled.local_names if name in private}))
    restored = dict(compiled.original.__globals__)
    restored.update(payload)
    restored = clone_environment(restored, compiled)
    assert restored["read_clock"].__globals__ is restored
    restored["i"] = 2
    assert restored["read_clock"]() == 2
    generator.close()


def test_metrics_include_discount_and_previous_physical_input():
    trace = {"reward": np.array([1., 2., 3.]), "y": np.array([[2., 4.], [3., 5.], [4., 6.]]),
             "reference": np.ones((3, 2)), "u": np.array([[2., 3.], [4., 2.], [3., 2.]]),
             "previous_input": np.array([1., 1.])}
    result = branch_metrics(trace, 2, .5)
    assert result["return"] == 2.
    np.testing.assert_allclose(result["sum_abs_input_movement_physical"], [3., 3.])
    np.testing.assert_allclose(result["max_abs_error_physical"], [2., 4.])


def test_summary_denominators_ties_and_single_seed_scope(tmp_path):
    records = []
    for selected, delta in [(2, -3.), (2, 1.), (1, 2.), (1, .05)]:
        records.append({"context": "standalone", "channel": "weights", "window": "release", "smoke": False,
                        "seed": 7, "category": "weights", "step": len(records), "status": "completed", "branches": {},
                        "continuation": "gated", "gate": {"selected_source": selected, "advantage_policy_supervisor": 1.},
                        "gate_config": {"advantage_margin": .5}, "metrics": {"20": {"delta_return": delta}}})
    rows = summarize(records, tmp_path, .1)
    assert rows[0]["harmful_acceptance_rate"] == .5
    assert rows[0]["beneficial_rejection_rate"] == .5
    assert rows[0]["training_seeds"] == 1
    assert rows[0]["ties"] == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "confidence" not in summary
    assert summary["training_seeds"] == [7]


def test_dry_run_creates_no_outputs(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(Path(cli.__file__)), "--dry-run", "--output-dir", str(tmp_path / "unused")],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "3200" in result.stdout
    assert not (tmp_path / "unused").exists()


def test_invalid_budget_fails_before_creating_output(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(Path(cli.__file__)), "--rollout-steps", "0",
                             "--output-dir", str(tmp_path / "unused")], text=True, capture_output=True)
    assert result.returncode != 0
    assert not (tmp_path / "unused").exists()


def test_combined_entrypoint_uses_gated_defaults_provider(monkeypatch, tmp_path):
    import utils.combined_runner as module
    captured = {}
    def stop_before_training(combined_cfg, runtime_ctx):
        captured.update(combined_cfg)
        return {}
    monkeypatch.setattr(module, "run_combined_supervisor", stop_before_training)
    cfg = lambda defaults: cli.configure_defaults(defaults, "combined", 7, tmp_path)
    namespace = cli.execute_entrypoint("combined", cfg)
    assert namespace["COMBINED_AGENT_MODE"] == "sg"
    assert captured["horizon_agent_kind"] == "sg_dqn"
    assert namespace["MARKOV_AGENT_KIND"] == "sg_td3"
    assert namespace["WEIGHTS_AGENT_KIND"] == "sg_td3"
    assert namespace["RESIDUAL_AGENT_KIND"] == "sg_td3"


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training_for_parity(category, output, *, capture, snapshots=0):
    """All 300 episodes, short technical dwell, original warm-up/hand-off rules.

    Small networks reduce test cost only; the study runner never alters widths.
    Every compared side uses exactly the same test configuration.
    """
    _seed(7)
    torch.set_num_threads(1)
    def configure(defaults):
        cfg = cli.configure_defaults(defaults, category, 7, output, smoke=True)
        for key in cli.SEED_KEYS:
            if isinstance(cfg.get(key), dict):
                if "batch_size" in cfg[key]:
                    cfg[key]["batch_size"] = 8
                for field in ("hidden_layers", "actor_hidden", "critic_hidden"):
                    if field in cfg[key]:
                        cfg[key][field] = [32, 32]
        return cfg
    with compressed_smoke_clock(category):
        if capture:
            options = StudyOptions(decisions_per_window=1, rollout_steps=8, horizons=(4, 8),
                                   long_rollout_steps=8, long_fraction=0., snapshots_per_window=snapshots, smoke=True)
            records = []
            factory = lambda compiled, args, kwargs: StudySession(compiled, output / "capture", options, records=records)
            with install_runner(category, factory):
                namespace = cli.execute_entrypoint(category, configure)
            assert len(records) == (16 if category == "combined" else 4)
            assert all(r["status"] == "completed" for r in records), [(r["step"], r["branches"]) for r in records]
        else:
            namespace = cli.execute_entrypoint(category, configure)
    result = namespace["result_bundle"]
    # Copy evidence before releasing the setup namespace/agents.
    evidence = {key: value.copy() for key, value in result.items() if isinstance(value, np.ndarray)}
    agents = []
    for value in namespace.values():
        if hasattr(value, "buffer") and hasattr(value, "train_steps"):
            agents.append(value)
    evidence["agent_steps"] = np.array([[a.steps, a.train_steps, a.buffer.size, a.buffer.ptr] for a in agents])
    for index, agent in enumerate(agents):
        for field in ("states", "actions", "rewards", "next_states"):
            evidence[f"replay_{index}_{field}"] = getattr(agent.buffer, field)[:agent.buffer.size].copy()
        for name, value in vars(agent).items():
            if isinstance(value, torch.nn.Module):
                for parameter, tensor in value.state_dict().items():
                    evidence[f"network_{index}_{name}_{parameter}"] = tensor.detach().cpu().numpy().copy()
    return evidence


@pytest.mark.parametrize("category", cli.ENTRIES)
def test_capture_preserves_original_full_training_and_handoff(category, tmp_path):
    reference = run_training_for_parity(category, tmp_path / "reference", capture=False)
    captured = run_training_for_parity(category, tmp_path / "captured", capture=True)
    assert set(reference) == set(captured)
    for key in reference:
        if reference[key].dtype.kind in "fciub":
            np.testing.assert_allclose(captured[key], reference[key], rtol=1e-8, atol=1e-9,
                                       equal_nan=True, err_msg=f"Training changed: {category}/{key}")


@pytest.mark.parametrize("category", ["weights", "markov", "combined"])
def test_saved_snapshot_reproduces_original_matched_branches(category, tmp_path):
    from review_experiments.e1.snapshots import replay_snapshot
    run_training_for_parity(category, tmp_path / "capture_run", capture=True, snapshots=1)
    captures = tmp_path / "capture_run" / "capture"
    records = [json.loads(line) for line in (captures / "decisions.jsonl").read_text().splitlines()]
    selected = next(r for r in records if r["window"] == "plant_change" and (category != "combined" or r["channel"] == "markov"))
    output = tmp_path / "replayed"
    results = replay_snapshot(captures / selected["snapshot"], output)
    assert all(r["state"] == "completed" for r in results.values())
    with np.load(captures / selected["trace"]) as expected:
        for source in ("policy", "supervisor"):
            with np.load(output / f"{source}.npz") as actual:
                for field in ("reward", "y", "u", "reference"):
                    np.testing.assert_allclose(actual[field], expected[f"{source}__{field}"], rtol=1e-8, atol=1e-9,
                                               err_msg=f"Snapshot mismatch: {category}/{source}/{field}")


def test_prediction_probe_uses_common_inputs_and_frozen_corrected_model():
    from types import SimpleNamespace
    from review_experiments.e1.prediction import prediction_probe
    from utils.markov_runner import compute_markov_blocks, make_markov_basis
    A, B, C = np.diag([.8, .7]), np.eye(2), np.eye(2)
    blocks = compute_markov_blocks(A, B, C, 9)
    basis, labels = make_markov_basis(blocks, "io_pair_gain")
    inputs = np.tile([.2, .3], (8, 1))
    outputs = []
    state = np.zeros(2)
    for u in inputs:
        state = A @ state + 1.2 * B @ u
        outputs.append(C @ state)
    snapshot = {"A": A, "C": C, "x_model": np.zeros(2), "m_blocks": blocks, "basis_blocks": basis,
                "config": {"z_bound": 1.}, "ctx": {"data_min": np.zeros(4), "data_max": np.ones(4),
                "ss_scaled_inputs": np.zeros(2), "y_ss_scaled": np.zeros(2)}}
    decision = SimpleNamespace(policy_action=np.full(4, .2), supervisor_action=np.full(4, .1))
    result, forecast = prediction_probe(snapshot, decision, {"u": inputs, "y": np.array(outputs)}, "markov")
    nominal = np.array(result["metrics"]["nominal"]["8"]["rmse_physical"])
    supervisor = np.array(result["metrics"]["supervisor_ls_or_zero"]["8"]["rmse_physical"])
    policy = np.array(result["metrics"]["rl_proposed"]["8"]["rmse_physical"])
    assert np.all(nominal > supervisor)
    np.testing.assert_allclose(policy, 0., atol=1e-12)
    np.testing.assert_allclose(forecast["rl_proposed"], outputs, atol=1e-12)
