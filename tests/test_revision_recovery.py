"""Recovery must preserve learning, not merely inference network parameters."""
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from revision_recovery.checkpoints import CheckpointManager, CheckpointRunner, CheckpointStop
from review_experiments.e1.runtime import rng_state, restore_rng


def training_fixture(configuration, runtime_ctx):
    agent = runtime_ctx["agent"]
    horizon_agent = runtime_ctx["horizon_agent"]
    nFE = 16
    time_in_sub_episodes = 4
    history = []
    state = np.zeros(3, dtype=np.float32)
    def read_clock():
        return i
    for i in range(nFE):
        weight_action_decision = agent.select_action_with_supervisor(state, np.zeros(2))
        next_state = np.asarray([random.random(), np.random.rand(), torch.rand(()).item()], dtype=np.float32)
        reward = float(np.sum(weight_action_decision.action))
        agent.push_supervised(state, weight_action_decision.action, reward, next_state, False)
        agent.train_step()
        horizon_agent.push(state, i % 4, reward, next_state, False)
        horizon_agent.train_step()
        history.append((read_clock(), reward, next_state.copy()))
        state = next_state
    return agent, horizon_agent, history


class Session:
    def __init__(self, output):
        self.output = output
        self.all_records = []
    def wants(self, key, env):
        return False
    def is_active(self, key):
        return False


def agents():
    from TD3Agent.supervisor_gated_agent import SupervisorGatedTD3Agent
    from DQN.dqn_agent import DQNAgent
    return {"agent": SupervisorGatedTD3Agent(3, 2, [8], [8], batch_size=4, buffer_size=32,
                                             device=torch.device("cpu"), seed=7),
            "horizon_agent": DQNAgent(3, 4, [8], batch_size=4, buffer_size=32,
                                       device=torch.device("cpu"), n_step=3, multistep_mode="n_step")}


def execute(directory, stop=None):
    compiled = CheckpointRunner(training_fixture, "weights")
    session = Session(directory / "capture")
    manager = CheckpointManager(directory / "checkpoints", {"fixture": "unchanged"},
                                every_cycles=1, stop_after_checkpoint=stop)
    manager.compiled = compiled
    env = compiled.environment(session, ({}, agents()), {})
    env.update(__recovery_manager=manager, __recovery_next_step=0)
    generator = env[training_fixture.__name__]()
    try:
        while True:
            kind, _ = next(generator)
            assert kind == "checkpoint"
            manager.save(env, session)
    except StopIteration as result:
        return result.value
    finally:
        generator.close()


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_checkpoint_resume_preserves_training_rng_optimizers_replay_and_pending_nstep(tmp_path):
    torch.set_num_threads(1)
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    state = rng_state()
    expected = execute(tmp_path / "continuous")
    expected_rng = rng_state()
    restore_rng(state)
    with pytest.raises(CheckpointStop):
        execute(tmp_path / "resumed", stop=8)
    actual = execute(tmp_path / "resumed")
    assert_nested_equal(actual[2], expected[2])
    assert_nested_equal(rng_state(), expected_rng)
    for observed, target in zip(actual[:2], expected[:2]):
        assert_nested_equal(observed.state_dict(), target.state_dict())
        assert observed.train_steps == target.train_steps > 0
        assert_nested_equal(observed.buffer.states, target.buffer.states)
        assert_nested_equal(observed.buffer.actions, target.buffer.actions)
        optimizers = [key for key, val in vars(target).items() if isinstance(val, torch.optim.Optimizer)]
        assert optimizers
        for key in optimizers:
            assert_nested_equal(getattr(observed, key).state_dict(), getattr(target, key).state_dict())
    assert len(actual[1].nstep_accumulator._buffer) == 2
    assert_nested_equal(list(actual[1].nstep_accumulator._buffer), list(expected[1].nstep_accumulator._buffer))


def test_checkpoint_guard_rejects_source_changes_and_corrupt_payload(tmp_path):
    with pytest.raises(CheckpointStop):
        execute(tmp_path, stop=4)
    manager = CheckpointManager(tmp_path / "checkpoints", {"fixture": "changed"})
    manager.compiled = CheckpointRunner(training_fixture, "weights")
    with pytest.raises(RuntimeError, match="dependencies differ"):
        manager.restore({}, Session(tmp_path / "capture"))
    manager.sources = {"fixture": "unchanged"}
    path = tmp_path / "checkpoints/training_state_0.pkl"
    path.write_bytes(path.read_bytes() + b"corrupted")
    with pytest.raises(RuntimeError, match="checksum"):
        manager.restore({}, Session(tmp_path / "capture"))


def test_resume_archives_post_checkpoint_decisions_and_orphan_files(tmp_path):
    capture = tmp_path / "capture"
    (capture / "traces").mkdir(parents=True)
    retained = {"step": 1, "trace": "traces/retained.npz"}
    extra = {"step": 2, "trace": "traces/replayed.npz"}
    for name in ("retained", "replayed", "orphan"):
        (capture / "traces" / (name + ".npz")).write_bytes(name.encode())
    (capture / "decisions.jsonl").write_text(json.dumps(retained) + "\n" + json.dumps(extra) + "\n")
    session = Session(capture)
    session.all_records = [retained]
    CheckpointManager._reconcile_capture(session)
    assert (capture / "decisions.jsonl").read_text() == json.dumps(retained) + "\n"
    assert sorted(path.name for path in (capture / "traces").iterdir()) == ["retained.npz"]
    archive = next((capture / "replayed_suffixes").iterdir())
    assert (archive / "traces/replayed.npz").read_bytes() == b"replayed"
    assert (archive / "traces/orphan.npz").read_bytes() == b"orphan"


def test_recovery_environment_is_process_local_and_preserves_original_settings():
    import os
    import run_polymer_remaining as recovery
    before = dict(os.environ)
    runtime = recovery.environment()
    assert runtime["NPY_DISABLE_CPU_FEATURES"] == "X86_V3,AVX2,FMA3"
    assert runtime["OPENBLAS_CORETYPE"] == "NEHALEM"
    assert runtime["OPENBLAS_NUM_THREADS"] == "1"
    assert dict(os.environ) == before
    from run_polymer_e1 import configure_defaults
    from systems.polymer import get_polymer_notebook_defaults
    original = get_polymer_notebook_defaults("combined")
    configured = configure_defaults(original, "combined", 7, Path("new_attempt"))
    assert configured["training_profile_override"] == "robustness_200_100"
    for key in ("warm_subepisodes", "warm_start_episodes", "post_warm_start_action_freeze_subepisodes"):
        if key in original:
            assert configured[key] == original[key]


@pytest.mark.parametrize("truncated", [False, True])
def test_resume_preserves_orphan_created_before_decision_log_append(tmp_path, truncated):
    capture = tmp_path / "capture"
    (capture / "traces").mkdir(parents=True)
    (capture / "traces/orphan.npz").write_bytes(b"partial trace")
    (capture / "decisions.jsonl").write_text('{"step":' if truncated else "")
    CheckpointManager._reconcile_capture(Session(capture))
    assert not (capture / "traces/orphan.npz").exists()
    assert (capture / "decisions.jsonl").read_text() == ""
    archive = next((capture / "replayed_suffixes").iterdir())
    assert (archive / "traces/orphan.npz").read_bytes() == b"partial trace"
