from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DuelingDQN.dueling_dqn_agent import DuelingDQNAgent
from systems.distillation import get_distillation_notebook_defaults
from systems.polymer import get_polymer_notebook_defaults


def test_polymer_discrete_defaults_use_epsilon_greedy():
    standard = get_polymer_notebook_defaults("horizon_standard")
    dueling = get_polymer_notebook_defaults("horizon_dueling")

    assert standard["agent"]["exploration_mode"] == "epsilon"
    assert dueling["agent"]["exploration_mode"] == "epsilon"


def test_distillation_discrete_defaults_use_epsilon_greedy():
    standard = get_distillation_notebook_defaults("horizon_standard")

    assert standard["agent"]["exploration_mode"] == "epsilon"


def test_polymer_td3_style_defaults_use_gaussian_noise():
    matrix = get_polymer_notebook_defaults("matrix")
    markov = get_polymer_notebook_defaults("markov")
    structured = get_polymer_notebook_defaults("structured_matrix")
    weights = get_polymer_notebook_defaults("weights")
    residual = get_polymer_notebook_defaults("residual")

    for nb in (matrix, markov, structured, weights, residual):
        assert nb["td3_agent"]["exploration_mode"] == "gaussian"


def test_distillation_active_td3_defaults_use_intended_noise():
    markov = get_distillation_notebook_defaults("markov")
    weights = get_distillation_notebook_defaults("weights")
    residual = get_distillation_notebook_defaults("residual")

    assert markov["td3_agent"]["exploration_mode"] == "param_noise"
    assert weights["td3_agent"]["exploration_mode"] == "gaussian"
    assert residual["td3_agent"]["exploration_mode"] == "param_noise"


def test_distillation_agent_defaults_use_40k_replay_buffer():
    families = (
        "horizon_standard",
        "markov",
        "weights",
        "residual",
    )
    agent_keys = ("agent", "td3_agent", "sac_agent")
    for family in families:
        nb = get_distillation_notebook_defaults(family)
        for agent_key in agent_keys:
            if agent_key in nb:
                assert nb[agent_key]["buffer_size"] == 40_000


def test_dueling_dqn_constructor_default_is_epsilon():
    agent = DuelingDQNAgent(
        state_dim=3,
        action_dim=2,
        hidden_dim=[],
        batch_size=4,
        buffer_size=16,
        device=torch.device("cpu"),
    )
    assert agent.exploration_mode == "epsilon"


def run_direct():
    test_polymer_discrete_defaults_use_epsilon_greedy()
    test_distillation_discrete_defaults_use_epsilon_greedy()
    test_polymer_td3_style_defaults_use_gaussian_noise()
    test_distillation_active_td3_defaults_use_intended_noise()
    test_distillation_agent_defaults_use_40k_replay_buffer()
    test_dueling_dqn_constructor_default_is_epsilon()
    print("exploration default tests passed")


if __name__ == "__main__":
    run_direct()
