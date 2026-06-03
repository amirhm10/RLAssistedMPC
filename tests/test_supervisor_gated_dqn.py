from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DQN.dqn_agent import DQNAgent
from DQN.supervisor_gated_dqn_agent import (
    DiscreteSupervisorGateConfig,
    SOURCE_POLICY,
    SOURCE_SUPERVISOR,
    SupervisorGatedDQNAgent,
)
from DuelingDQN import SupervisorGatedDuelingDQNAgent
from DuelingDQN.dueling_dqn_agent import DuelingDQNAgent


def make_sg_dqn(**overrides):
    kwargs = {
        "state_dim": 3,
        "action_dim": 3,
        "hidden_dim": [],
        "batch_size": 8,
        "buffer_size": 64,
        "device": torch.device("cpu"),
        "exploration_mode": "epsilon",
        "eps_start": 0.0,
        "eps_end": 0.0,
        "eps_decay_mode": "linear",
        "multistep_mode": "one_step",
        "n_step": 1,
    }
    kwargs.update(overrides)
    return SupervisorGatedDQNAgent(**kwargs)


def make_sg_dueling(**overrides):
    kwargs = {
        "state_dim": 3,
        "action_dim": 3,
        "hidden_dim": [],
        "batch_size": 8,
        "buffer_size": 64,
        "device": torch.device("cpu"),
        "exploration_mode": "epsilon",
        "eps_start": 0.0,
        "eps_end": 0.0,
        "eps_decay_mode": "linear",
        "multistep_mode": "n_step",
        "n_step": 1,
    }
    kwargs.update(overrides)
    return SupervisorGatedDuelingDQNAgent(**kwargs)


def set_dqn_q_values(agent, values):
    values_t = torch.as_tensor(values, dtype=torch.float32)
    for network in (agent.online, agent.target):
        layer = network.network[-1]
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.copy_(values_t)


def set_dueling_q_values(agent, values):
    values_t = torch.as_tensor(values, dtype=torch.float32)
    for network in (agent.online, agent.target):
        with torch.no_grad():
            network.value_head.weight.zero_()
            network.value_head.bias.zero_()
            network.advantage_head.weight.zero_()
            network.advantage_head.bias.copy_(values_t)


def random_transition(i):
    rng = np.random.default_rng(i)
    state = rng.normal(size=3).astype(np.float32)
    next_state = (state + 0.1 * rng.normal(size=3)).astype(np.float32)
    reward = float(rng.normal())
    done = bool(i % 17 == 0 and i > 0)
    return state, reward, next_state, done


def test_import_supervisor_gated_dqn_classes():
    assert SupervisorGatedDQNAgent is not None
    assert SupervisorGatedDuelingDQNAgent is not None
    assert DiscreteSupervisorGateConfig is not None


def test_tie_defaults_to_supervisor():
    agent = make_sg_dqn(
        supervisor_gate_config=DiscreteSupervisorGateConfig(
            advantage_margin=0.0,
            default_to_supervisor=True,
        )
    )
    set_dqn_q_values(agent, [0.0, 1.0, 0.5])
    decision = agent.select_action_with_supervisor(
        np.zeros(3, dtype=np.float32),
        supervisor_action=1,
        explore=False,
    )
    assert decision.policy_action == 1
    assert decision.action == 1
    assert decision.selected_source == SOURCE_SUPERVISOR
    assert decision.advantage_policy_supervisor == 0.0


def test_policy_selected_when_q_advantage_exceeds_margin():
    agent = make_sg_dqn(
        supervisor_gate_config=DiscreteSupervisorGateConfig(
            advantage_margin=0.0,
            default_to_supervisor=True,
        )
    )
    set_dqn_q_values(agent, [0.0, 1.0, 3.0])
    decision = agent.select_action_with_supervisor(
        np.zeros(3, dtype=np.float32),
        supervisor_action=0,
        explore=False,
    )
    assert decision.policy_action == 2
    assert decision.action == 2
    assert decision.selected_source == SOURCE_POLICY
    assert decision.score_policy > decision.score_supervisor


def test_supervisor_selected_when_margin_not_met():
    agent = make_sg_dqn(
        supervisor_gate_config=DiscreteSupervisorGateConfig(
            advantage_margin=0.5,
            default_to_supervisor=True,
        )
    )
    set_dqn_q_values(agent, [0.0, 0.1, 0.2])
    decision = agent.select_action_with_supervisor(
        np.zeros(3, dtype=np.float32),
        supervisor_action=1,
        explore=False,
    )
    assert decision.policy_action == 2
    assert decision.action == 1
    assert decision.selected_source == SOURCE_SUPERVISOR
    assert 0.0 < decision.advantage_policy_supervisor < 0.5


def test_min_train_steps_blocks_policy_gate():
    agent = make_sg_dqn(
        supervisor_gate_config=DiscreteSupervisorGateConfig(
            advantage_margin=0.0,
            default_to_supervisor=True,
            min_train_steps_before_policy_gate=10,
        )
    )
    set_dqn_q_values(agent, [0.0, 1.0, 3.0])
    decision = agent.select_action_with_supervisor(
        np.zeros(3, dtype=np.float32),
        supervisor_action=0,
        explore=False,
    )
    assert decision.policy_action == 2
    assert decision.action == 0
    assert decision.selected_source == SOURCE_SUPERVISOR


def test_replay_metadata_roundtrip():
    agent = make_sg_dqn()
    state, reward, next_state, done = random_transition(1)
    agent.push_supervised(
        state,
        2,
        reward,
        next_state,
        done,
        policy_action=2,
        supervisor_action=0,
        previous_action=1,
        selected_source=SOURCE_POLICY,
        score_policy=3.0,
        score_supervisor=1.0,
        advantage_policy_supervisor=2.0,
    )
    batch = agent.buffer.sample_supervised(1, device=agent.device)
    assert batch["policy_actions"].shape == (1,)
    assert int(batch["actions"][0].item()) == 2
    assert int(batch["policy_actions"][0].item()) == 2
    assert int(batch["supervisor_actions"][0].item()) == 0
    assert int(batch["previous_actions"][0].item()) == 1
    assert int(batch["selected_sources"][0].item()) == SOURCE_POLICY
    assert float(batch["advantage_policy_supervisor"][0].item()) == 2.0


def test_sg_dqn_one_step_training_smoke():
    agent = make_sg_dqn()
    for i in range(16):
        state, reward, next_state, done = random_transition(i)
        executed = i % agent.action_dim
        agent.push_supervised(
            state,
            executed,
            reward,
            next_state,
            done,
            policy_action=executed,
            supervisor_action=0,
            previous_action=0,
            selected_source=SOURCE_POLICY if i % 2 else SOURCE_SUPERVISOR,
            score_policy=float(i),
            score_supervisor=float(i) - 0.5,
            advantage_policy_supervisor=0.5,
        )
    loss = agent.train_step()
    assert isinstance(loss, float)
    assert len(agent.loss_history) == 1


def test_sg_dueling_n_step_one_training_smoke():
    agent = make_sg_dueling()
    set_dueling_q_values(agent, [0.0, 1.0, 2.0])
    decision = agent.select_action_with_supervisor(
        np.zeros(3, dtype=np.float32),
        supervisor_action=0,
        explore=False,
    )
    assert decision.action == 2
    for i in range(16):
        state, reward, next_state, done = random_transition(100 + i)
        executed = i % agent.action_dim
        agent.push_supervised(
            state,
            executed,
            reward,
            next_state,
            done,
            policy_action=executed,
            supervisor_action=0,
            previous_action=0,
            selected_source=SOURCE_POLICY if i % 2 else SOURCE_SUPERVISOR,
            score_policy=float(i),
            score_supervisor=float(i) - 0.5,
            advantage_policy_supervisor=0.5,
        )
    loss = agent.train_step()
    assert isinstance(loss, float)
    assert len(agent.loss_history) == 1
    assert len(agent.avg_value_trace) == 1
    assert len(agent.avg_advantage_spread_trace) == 1


def test_original_dqn_classes_remain_usable():
    dqn = DQNAgent(
        state_dim=3,
        action_dim=3,
        hidden_dim=[],
        batch_size=4,
        buffer_size=16,
        device=torch.device("cpu"),
        exploration_mode="epsilon",
        eps_start=0.0,
        eps_end=0.0,
    )
    dueling = DuelingDQNAgent(
        state_dim=3,
        action_dim=3,
        hidden_dim=[],
        batch_size=4,
        buffer_size=16,
        device=torch.device("cpu"),
        exploration_mode="epsilon",
        eps_start=0.0,
        eps_end=0.0,
        multistep_mode="n_step",
        n_step=1,
    )
    state = np.zeros(3, dtype=np.float32)
    assert 0 <= dqn.take_action(state) < 3
    assert 0 <= dueling.take_action(state) < 3


def test_true_multistep_modes_raise_clear_error():
    for factory, kwargs in [
        (make_sg_dqn, {"multistep_mode": "n_step", "n_step": 2}),
        (make_sg_dqn, {"multistep_mode": "lambda", "n_step": 2}),
        (make_sg_dueling, {"multistep_mode": "retrace", "n_step": 2}),
    ]:
        try:
            factory(**kwargs)
        except NotImplementedError as exc:
            assert "Supervisor-gated DQN v1" in str(exc)
        else:
            raise AssertionError("Expected NotImplementedError for unsupported SG-DQN mode.")


def run_direct():
    test_import_supervisor_gated_dqn_classes()
    test_tie_defaults_to_supervisor()
    test_policy_selected_when_q_advantage_exceeds_margin()
    test_supervisor_selected_when_margin_not_met()
    test_min_train_steps_blocks_policy_gate()
    test_replay_metadata_roundtrip()
    test_sg_dqn_one_step_training_smoke()
    test_sg_dueling_n_step_one_training_smoke()
    test_original_dqn_classes_remain_usable()
    test_true_multistep_modes_raise_clear_error()
    print("supervisor_gated_dqn tests passed")


if __name__ == "__main__":
    run_direct()
