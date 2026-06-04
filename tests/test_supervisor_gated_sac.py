from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SACAgent.sac_agent import SACAgent
from SACAgent.supervisor_gated_sac_agent import (
    SACSupervisorGateConfig,
    SupervisorGatedSACAgent,
)
from TD3Agent.supervisor_replay_buffer import SOURCE_POLICY, SOURCE_SUPERVISOR


def make_agent(**overrides):
    kwargs = {
        "state_dim": 3,
        "action_dim": 2,
        "actor_hidden": [],
        "critic_hidden": [],
        "batch_size": 8,
        "buffer_size": 64,
        "device": torch.device("cpu"),
        "actor_freeze": 0,
        "alpha_freeze": "actor_freeze",
        "actor_q_mode": "min",
        "init_alpha": 0.05,
        "learn_alpha": True,
        "target_entropy": -2.0,
        "multistep_mode": "one_step",
        "n_step": 1,
    }
    kwargs.update(overrides)
    return SupervisorGatedSACAgent(**kwargs)


def random_transition(i):
    rng = np.random.default_rng(i)
    state = rng.normal(size=3).astype(np.float32)
    next_state = (state + 0.1 * rng.normal(size=3)).astype(np.float32)
    action = rng.uniform(-0.6, 0.6, size=2).astype(np.float32)
    supervisor = rng.uniform(-0.25, 0.25, size=2).astype(np.float32)
    previous = rng.uniform(-0.25, 0.25, size=2).astype(np.float32)
    reward = float(rng.normal())
    done = bool(i % 23 == 0 and i > 0)
    return state, action, reward, next_state, done, supervisor, previous


def set_actor_mean_action(agent, action):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    clipped = np.clip(action / agent.max_action, -0.999, 0.999)
    pre_tanh = np.arctanh(clipped).astype(np.float32)
    with torch.no_grad():
        for param in agent.actor.parameters():
            param.zero_()
        agent.actor.mean_layer.bias.copy_(torch.as_tensor(pre_tanh))
        agent.actor.log_std_layer.bias.fill_(-20.0)


def set_linear_action_q(agent, weights, bias=0.0):
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    assert weights.size == agent.buffer.action_dim
    for network_name in ("q1_network", "q2_network"):
        network = getattr(agent.critic, network_name)
        layer = network[-1]
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.fill_(float(bias))
            layer.weight[0, agent.buffer.state_dim : agent.buffer.state_dim + agent.buffer.action_dim] = (
                torch.as_tensor(weights)
            )


def set_twin_linear_action_q(agent, q1_weights, q2_weights, q1_bias=0.0, q2_bias=0.0):
    q_specs = {
        "q1_network": (q1_weights, q1_bias),
        "q2_network": (q2_weights, q2_bias),
    }
    for network_name, (weights, bias) in q_specs.items():
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        assert weights.size == agent.buffer.action_dim
        network = getattr(agent.critic, network_name)
        layer = network[-1]
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.fill_(float(bias))
            layer.weight[0, agent.buffer.state_dim : agent.buffer.state_dim + agent.buffer.action_dim] = (
                torch.as_tensor(weights)
            )


def test_import_supervisor_gated_sac_classes():
    assert SupervisorGatedSACAgent is not None
    assert SACSupervisorGateConfig is not None


def test_tie_defaults_to_supervisor():
    agent = make_agent(
        supervisor_gate_config=SACSupervisorGateConfig(
            advantage_margin=0.0,
            default_to_supervisor=True,
        )
    )
    set_actor_mean_action(agent, [0.0, 0.0])
    set_linear_action_q(agent, [1.0, 0.0])
    decision = agent.select_action_with_supervisor(
        np.zeros(3, dtype=np.float32),
        supervisor_action=np.zeros(2, dtype=np.float32),
        explore=False,
    )
    assert decision.selected_source == SOURCE_SUPERVISOR
    assert np.allclose(decision.action, np.zeros(2))
    assert decision.advantage_policy_supervisor == 0.0


def test_policy_selected_when_q_advantage_exceeds_margin():
    agent = make_agent(
        supervisor_gate_config=SACSupervisorGateConfig(
            advantage_margin=0.0,
            default_to_supervisor=True,
        )
    )
    set_actor_mean_action(agent, [0.5, 0.0])
    set_linear_action_q(agent, [1.0, 0.0])
    decision = agent.select_action_with_supervisor(
        np.zeros(3, dtype=np.float32),
        supervisor_action=np.array([-0.5, 0.0], dtype=np.float32),
        explore=False,
    )
    assert decision.selected_source == SOURCE_POLICY
    assert np.allclose(decision.action, decision.policy_action)
    assert decision.score_policy > decision.score_supervisor


def test_supervisor_selected_when_margin_not_met():
    agent = make_agent(
        supervisor_gate_config=SACSupervisorGateConfig(
            advantage_margin=0.5,
            default_to_supervisor=True,
        )
    )
    set_actor_mean_action(agent, [0.1, 0.0])
    set_linear_action_q(agent, [1.0, 0.0])
    decision = agent.select_action_with_supervisor(
        np.zeros(3, dtype=np.float32),
        supervisor_action=np.zeros(2, dtype=np.float32),
        explore=False,
    )
    assert decision.selected_source == SOURCE_SUPERVISOR
    assert 0.0 < decision.advantage_policy_supervisor < 0.5


def test_deterministic_candidate_mode_ignores_sampled_action():
    agent = make_agent(
        supervisor_gate_config=SACSupervisorGateConfig(
            candidate_mode="deterministic",
            advantage_margin=0.0,
            default_to_supervisor=True,
        )
    )
    set_actor_mean_action(agent, [0.5, 0.0])
    set_linear_action_q(agent, [1.0, 0.0])

    def fail_sample(_state):
        raise AssertionError("deterministic SG-SAC gate should not sample the actor")

    agent.actor.sample = fail_sample
    decision = agent.select_action_with_supervisor(
        np.zeros(3, dtype=np.float32),
        supervisor_action=np.zeros(2, dtype=np.float32),
        explore=True,
        test=False,
    )
    assert decision.selected_source == SOURCE_POLICY
    assert np.allclose(decision.policy_action, np.array([0.5, 0.0], dtype=np.float32), atol=1e-6)
    assert agent.last_exploration_value == 0.0


def test_critic_dominance_gate_rejects_one_critic_deficit():
    agent = make_agent(
        supervisor_gate_config=SACSupervisorGateConfig(
            advantage_margin=0.0,
            default_to_supervisor=True,
            score_uncertainty_weight=0.0,
            score_supervisor_action_weight=0.0,
            score_previous_action_weight=0.0,
            critic_dominance_gate_enabled=True,
            critic_dominance_margin=0.0,
        )
    )
    set_actor_mean_action(agent, [0.5, 0.0])
    set_twin_linear_action_q(
        agent,
        q1_weights=[20.0, 0.0],
        q1_bias=0.0,
        q2_weights=[-2.0, 0.0],
        q2_bias=2.0,
    )
    decision = agent.select_action_with_supervisor(
        np.zeros(3, dtype=np.float32),
        supervisor_action=np.zeros(2, dtype=np.float32),
        explore=False,
    )
    assert decision.score_policy > decision.score_supervisor
    assert decision.q1_policy > decision.q1_supervisor
    assert decision.q2_policy < decision.q2_supervisor
    assert decision.selected_source == SOURCE_SUPERVISOR


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
        score_supervisor=0.25,
        advantage_policy_supervisor=0.75,
    )
    batch = agent.buffer.sample_supervised(1, device=agent.device)
    assert batch["policy_actions"].shape == (1, 2)
    assert batch["supervisor_actions"].shape == (1, 2)
    assert batch["previous_actions"].shape == (1, 2)
    assert int(batch["selected_sources"][0].item()) == SOURCE_POLICY
    assert float(batch["advantage_policy_supervisor"][0].item()) == 0.75


def test_one_step_training_smoke_with_sg_bc_and_smoothness():
    agent = make_agent(
        supervisor_gate_config=SACSupervisorGateConfig(
            supervisor_bc_weight=0.1,
            sampled_supervisor_bc_weight=0.1,
            smooth_action_weight=0.05,
            enable_supervisor_actor_loss=True,
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
    assert meta["actor_updated"] is True
    assert meta["alpha_updated"] is True
    assert "sg_advantage" in meta
    assert np.isfinite(meta["sg_bc_loss"])
    assert np.isfinite(meta["sg_sampled_bc_loss"])
    assert np.isfinite(meta["sg_smooth_loss"])
    assert np.isfinite(agent.sg_sampled_bc_loss_trace[-1])


def test_existing_sac_agent_still_usable():
    agent = SACAgent(
        state_dim=3,
        action_dim=2,
        actor_hidden=[],
        critic_hidden=[],
        batch_size=8,
        buffer_size=64,
        device=torch.device("cpu"),
    )
    action = agent.take_action(np.zeros(3, dtype=np.float32), explore=True)
    assert action.shape == (2,)


def test_unsupported_multistep_modes_raise():
    for mode in ("n_step", "sac_n", "lambda"):
        try:
            make_agent(multistep_mode=mode, n_step=2)
        except NotImplementedError as exc:
            assert "one_step" in str(exc)
        else:
            raise AssertionError(f"Expected NotImplementedError for {mode}")


def run_direct():
    test_import_supervisor_gated_sac_classes()
    test_tie_defaults_to_supervisor()
    test_policy_selected_when_q_advantage_exceeds_margin()
    test_supervisor_selected_when_margin_not_met()
    test_deterministic_candidate_mode_ignores_sampled_action()
    test_critic_dominance_gate_rejects_one_critic_deficit()
    test_replay_metadata_roundtrip()
    test_one_step_training_smoke_with_sg_bc_and_smoothness()
    test_existing_sac_agent_still_usable()
    test_unsupported_multistep_modes_raise()
    print("supervisor_gated_sac tests passed")


if __name__ == "__main__":
    run_direct()
