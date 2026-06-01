from __future__ import annotations

import inspect
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TD3Agent.agent import TD3Agent
from TD3Agent.supervisor_gated_agent import SupervisorGatedTD3Agent, SupervisorGateConfig
from TD3Agent.supervisor_replay_buffer import SOURCE_POLICY, SOURCE_SUPERVISOR


def make_agent(**overrides):
    kwargs = {
        "state_dim": 4,
        "action_dim": 2,
        "actor_hidden": [8],
        "critic_hidden": [8],
        "batch_size": 8,
        "buffer_size": 64,
        "device": torch.device("cpu"),
        "seed": 7,
    }
    kwargs.update(overrides)
    return SupervisorGatedTD3Agent(**kwargs)


def random_transition(i):
    rng = np.random.default_rng(i)
    state = rng.normal(size=4).astype(np.float32)
    next_state = (state + 0.1 * rng.normal(size=4)).astype(np.float32)
    action = rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
    supervisor = rng.uniform(-0.25, 0.25, size=2).astype(np.float32)
    previous = rng.uniform(-0.25, 0.25, size=2).astype(np.float32)
    reward = float(rng.normal())
    return state, action, reward, next_state, False, supervisor, previous


def test_import_supervisor_gated_td3():
    assert SupervisorGatedTD3Agent is not None


def test_select_action_with_supervisor_shape():
    agent = make_agent()
    state = np.zeros(4, dtype=np.float32)
    supervisor = np.zeros(2, dtype=np.float32)
    decision = agent.select_action_with_supervisor(state, supervisor)
    assert decision.action.shape == (2,)
    assert decision.policy_action.shape == (2,)
    assert decision.supervisor_action.shape == (2,)


def test_default_to_supervisor_on_tie():
    agent = make_agent(
        supervisor_gate_config=SupervisorGateConfig(
            advantage_margin=0.0,
            default_to_supervisor=True,
        )
    )
    state = np.zeros(4, dtype=np.float32)
    supervisor = agent.act_eval(state).reshape(-1)
    decision = agent.select_action_with_supervisor(state, supervisor)
    assert decision.selected_source == SOURCE_SUPERVISOR
    assert np.allclose(decision.action, supervisor)


def test_replay_metadata_roundtrip():
    agent = make_agent()
    state, action, reward, next_state, done, supervisor, previous = random_transition(1)
    agent.push_supervised(
        state,
        action,
        reward,
        next_state,
        done,
        policy_action=action,
        supervisor_action=supervisor,
        previous_action=previous,
        selected_source=SOURCE_POLICY,
        score_policy=1.0,
        score_supervisor=0.5,
        advantage_policy_supervisor=0.5,
    )
    batch = agent.buffer.sample_supervised(1, device=agent.device)
    assert batch["policy_actions"].shape == (1, 2)
    assert batch["supervisor_actions"].shape == (1, 2)
    assert batch["previous_actions"].shape == (1, 2)
    assert int(batch["selected_sources"][0].item()) == SOURCE_POLICY


def test_training_smoke():
    agent = make_agent(
        supervisor_gate_config=SupervisorGateConfig(
            supervisor_bc_weight=0.01,
            smooth_action_weight=0.01,
        )
    )
    for i in range(16):
        state, action, reward, next_state, done, supervisor, previous = random_transition(i)
        agent.push_supervised(
            state,
            action,
            reward,
            next_state,
            done,
            policy_action=action,
            supervisor_action=supervisor,
            previous_action=previous,
            selected_source=SOURCE_POLICY if i % 2 else SOURCE_SUPERVISOR,
            score_policy=float(i),
            score_supervisor=float(i) - 0.25,
            advantage_policy_supervisor=0.25,
        )
    meta = agent.train_step()
    assert isinstance(meta, dict)
    assert meta["critic_updated"] is True
    assert "sg_advantage" in meta


def test_original_td3_train_step_signature_untouched():
    signature = inspect.signature(TD3Agent.train_step)
    assert "bc_context" in signature.parameters
    assert signature.parameters["bc_context"].default is None
    agent = TD3Agent(
        state_dim=4,
        action_dim=2,
        actor_hidden=[8],
        critic_hidden=[8],
        batch_size=8,
        buffer_size=64,
        device=torch.device("cpu"),
        seed=11,
    )
    assert hasattr(agent, "train_step")


def run_direct():
    test_import_supervisor_gated_td3()
    test_select_action_with_supervisor_shape()
    test_default_to_supervisor_on_tie()
    test_replay_metadata_roundtrip()
    test_training_smoke()
    test_original_td3_train_step_signature_untouched()
    print("supervisor_gated_td3 tests passed")


if __name__ == "__main__":
    run_direct()
