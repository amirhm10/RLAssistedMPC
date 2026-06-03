from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SACAgent.gaussian_actor import GaussianActor
from SACAgent.sac_agent import SACAgent


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
    }
    kwargs.update(overrides)
    return SACAgent(**kwargs)


def random_transition(i):
    rng = np.random.default_rng(i)
    state = rng.normal(size=3).astype(np.float32)
    next_state = (state + 0.1 * rng.normal(size=3)).astype(np.float32)
    action = rng.uniform(-0.7, 0.7, size=2).astype(np.float32)
    reward = float(rng.normal())
    done = bool(i % 19 == 0 and i > 0)
    return state, action, reward, next_state, done


def fill_buffer(agent, n=16):
    for i in range(n):
        state, action, reward, next_state, done = random_transition(i)
        agent.push(state, action, reward, next_state, done)


def test_sac_import_and_constructor():
    agent = make_agent()
    assert isinstance(agent, SACAgent)
    assert agent.actor_q_mode == "min"
    assert agent.alpha_freeze == agent.actor_freeze


def test_stochastic_training_action_and_deterministic_eval_action():
    agent = make_agent()
    state = np.zeros(3, dtype=np.float32)
    torch.manual_seed(1)
    train_action = agent.take_action(state, explore=True)
    eval_action = agent.act_eval(state)
    assert train_action.shape == (2,)
    assert eval_action.shape == (2,)
    assert np.max(np.abs(train_action)) <= agent.max_action + 1e-6
    assert np.max(np.abs(eval_action)) <= agent.max_action + 1e-6
    assert not np.allclose(train_action, eval_action)


def test_actor_loss_uses_configured_q_mode():
    agent = make_agent(actor_q_mode="mean")
    fill_buffer(agent)
    modes = []
    original = agent.critic.combined_forward

    def wrapped_combined_forward(state, action, mode="min"):
        modes.append(mode)
        return original(state, action, mode=mode)

    agent.critic.combined_forward = wrapped_combined_forward
    meta = agent.train_step()
    assert isinstance(meta, dict)
    assert meta["actor_q_mode"] == "mean"
    assert modes[-1] == "mean"


def test_alpha_freezes_then_updates():
    agent = make_agent(actor_freeze=2, alpha_freeze="actor_freeze", alpha_lr=1e-2)
    fill_buffer(agent)
    initial_log_alpha = float(agent.log_alpha.detach().item())
    meta0 = agent.train_step()
    meta1 = agent.train_step()
    assert meta0["alpha_updated"] is False
    assert meta1["alpha_updated"] is False
    assert float(agent.log_alpha.detach().item()) == initial_log_alpha
    meta2 = agent.train_step()
    assert meta2["alpha_updated"] is True
    assert float(agent.log_alpha.detach().item()) != initial_log_alpha


def test_fixed_alpha_does_not_update():
    agent = make_agent(learn_alpha=False, alpha_freeze=0, alpha_lr=1e-2)
    fill_buffer(agent)
    initial_log_alpha = float(agent.log_alpha.detach().item())
    meta = agent.train_step()
    assert meta["alpha_updated"] is False
    assert float(agent.log_alpha.detach().item()) == initial_log_alpha


def test_tanh_gaussian_log_prob_includes_max_action_scale():
    torch.manual_seed(4)
    actor_unit = GaussianActor(3, 2, [], max_action=1.0)
    actor_scaled = GaussianActor(3, 2, [], max_action=2.0)
    actor_scaled.load_state_dict(actor_unit.state_dict())
    state = torch.zeros(5, 3)

    torch.manual_seed(7)
    action_unit, logp_unit, _ = actor_unit.sample(state)
    torch.manual_seed(7)
    action_scaled, logp_scaled, _ = actor_scaled.sample(state)

    assert torch.allclose(action_scaled, 2.0 * action_unit, atol=1e-6)
    expected = logp_unit - 2.0 * math.log(2.0)
    assert torch.allclose(logp_scaled, expected, atol=1e-6)


def test_one_step_training_smoke_returns_sac_metadata():
    agent = make_agent()
    fill_buffer(agent)
    meta = agent.train_step()
    assert isinstance(meta, dict)
    assert meta["critic_updated"] is True
    assert meta["actor_updated"] is True
    assert meta["alpha_updated"] is True
    assert "alpha_loss" in meta
    assert "entropy" in meta
    assert len(agent.actor_losses) == 1
    assert len(agent.alpha_updated_trace) == 1


def run_direct():
    test_sac_import_and_constructor()
    test_stochastic_training_action_and_deterministic_eval_action()
    test_actor_loss_uses_configured_q_mode()
    test_alpha_freezes_then_updates()
    test_fixed_alpha_does_not_update()
    test_tanh_gaussian_log_prob_includes_max_action_scale()
    test_one_step_training_smoke_returns_sac_metadata()
    print("sac_agent_core tests passed")


if __name__ == "__main__":
    run_direct()
