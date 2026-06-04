from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DQN.supervisor_gated_dqn_agent import (
    SOURCE_HELD,
    SOURCE_POLICY,
    SOURCE_SUPERVISOR,
    SupervisorGatedDQNAgent,
)
from DuelingDQN.supervisor_gated_dueling_dqn_agent import SupervisorGatedDuelingDQNAgent
from systems.distillation import get_distillation_notebook_defaults
from systems.polymer import get_polymer_notebook_defaults
from utils.agent_step_runtime import (
    replay_train_supervisor_gated_horizon_agent,
    select_supervisor_gated_horizon_action,
)
from utils.phase1_hidden_release import ACTION_SOURCE_HELD_INTERVAL


def make_sg_dqn():
    return SupervisorGatedDQNAgent(
        state_dim=3,
        action_dim=3,
        hidden_dim=[],
        batch_size=4,
        buffer_size=32,
        device=torch.device("cpu"),
        exploration_mode="epsilon",
        eps_start=0.0,
        eps_end=0.0,
        eps_decay_mode="linear",
        multistep_mode="one_step",
        n_step=1,
        supervisor_gate_config={
            "advantage_margin": 0.0,
            "default_to_supervisor": True,
            "min_train_steps_before_policy_gate": 0,
        },
    )


def set_dqn_q_values(agent, values):
    values_t = torch.as_tensor(values, dtype=torch.float32)
    for network in (agent.online, agent.target):
        layer = network.network[-1]
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.copy_(values_t)


def test_wrapper_configs_set_sg_defaults_and_disable_old_safety():
    from distillation_RL_assisted_MPC_horizons_supervisor_gated_dqn_unified import (
        configure_sg_dqn_horizon_critic_warm,
    )
    from distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified import (
        configure_sg_dueling_dqn_horizon_critic_warm,
    )

    standard = configure_sg_dqn_horizon_critic_warm(get_distillation_notebook_defaults("horizon_standard"))
    dueling = configure_sg_dueling_dqn_horizon_critic_warm(get_distillation_notebook_defaults("horizon_dueling"))

    assert standard["agent_kind"] == "sg_dqn"
    assert dueling["agent_kind"] == "sg_dueling_dqn"
    for configured in (standard, dueling):
        assert configured["run_mode"] == "disturb"
        assert configured["disturbance_profile"] == "fluctuation"
        assert configured["state_mode"] == "standard"
        assert "standard" in configured["result_prefix_override"]
        assert "standard" in configured["compare_prefix_override"]
        assert configured["warm_start_override"] == 10
        assert configured["post_warm_start_action_freeze_subepisodes"] == 3
        assert configured["supervisor_gate"]["advantage_margin"] == 0.0
        assert configured["supervisor_gate"]["default_to_supervisor"] is True
        assert configured["supervisor_gate"]["min_train_steps_before_policy_gate"] == 0
        assert configured["agent"]["supervisor_gate"] == configured["supervisor_gate"]
        assert configured["agent"]["exploration_mode"] == "epsilon"
        assert configured["agent"]["eps_start"] == 0.2
        assert configured["agent"]["eps_end"] == 0.02
        assert configured["agent"]["eps_decay_steps"] == 18_600
        safety = configured["horizon_safety"]
        assert safety["enabled"] is False
        assert safety["release_filter"]["enabled"] is False
        assert safety["reward_probation"]["enabled"] is False
        assert safety["shadow_default_mpc"]["enabled"] is False


def test_distillation_dueling_aspen6_legacy_reward_wrapper_config():
    from distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_aspen6_legacy_reward_unified import (
        configure_sg_dueling_dqn_horizon_aspen6_legacy_reward,
    )

    configured = configure_sg_dueling_dqn_horizon_aspen6_legacy_reward(
        get_distillation_notebook_defaults("horizon_dueling")
    )

    assert configured["agent_kind"] == "sg_dueling_dqn"
    assert configured["run_mode"] == "disturb"
    assert configured["disturbance_profile"] == "fluctuation"
    assert configured["state_mode"] == "standard"
    assert configured["aspen_preset"] == 6
    assert "aspen6_legacyreward" in configured["result_prefix_override"]
    assert "aspen6_legacyreward" in configured["compare_prefix_override"]
    assert "standard" in configured["result_prefix_override"]
    assert "standard" in configured["compare_prefix_override"]

    reward = configured["reward"]
    np.testing.assert_allclose(reward["k_rel"], np.asarray([0.3, 0.02]))
    np.testing.assert_allclose(reward["band_floor_phys"], np.asarray([0.003, 0.3]))
    np.testing.assert_allclose(reward["Q_diag"], np.asarray([3.7e4, 1.5e3]))
    np.testing.assert_allclose(reward["R_diag"], np.asarray([2.5e3, 2.5e3]))
    assert reward["beta"] == 7.0
    assert reward["reward_scale"] == 1.0

    safety = configured["horizon_safety"]
    assert safety["enabled"] is False
    assert safety["release_filter"]["enabled"] is False
    assert safety["reward_probation"]["enabled"] is False
    assert safety["shadow_default_mpc"]["enabled"] is False


def test_wrapper_modules_reference_sg_agent_classes():
    import distillation_RL_assisted_MPC_horizons_supervisor_gated_dqn_unified as standard
    import distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified as dueling

    assert standard.SupervisorGatedDQNAgent is SupervisorGatedDQNAgent
    assert dueling.SupervisorGatedDuelingDQNAgent is SupervisorGatedDuelingDQNAgent


def test_polymer_wrapper_configs_set_sg_defaults_and_disable_old_safety():
    from RL_assisted_MPC_horizons_supervisor_gated_dqn_unified import (
        configure_sg_dqn_horizon_critic_warm,
    )
    from RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified import (
        configure_sg_dueling_dqn_horizon_critic_warm,
    )

    standard = configure_sg_dqn_horizon_critic_warm(get_polymer_notebook_defaults("horizon_standard"))
    dueling = configure_sg_dueling_dqn_horizon_critic_warm(get_polymer_notebook_defaults("horizon_dueling"))

    assert standard["agent_kind"] == "sg_dqn"
    assert dueling["agent_kind"] == "sg_dueling_dqn"
    for configured in (standard, dueling):
        assert configured["run_mode"] == "disturb"
        assert configured["state_mode"] == "standard"
        assert configured["warm_start_override"] == 10
        assert configured["post_warm_start_action_freeze_subepisodes"] == 3
        assert "standard" in configured["result_prefix_override"]
        assert "standard" in configured["compare_prefix_override"]
        assert configured["controller"]["predict_h"] == 9
        assert configured["controller"]["cont_h"] == 3
        assert configured["supervisor_gate"]["advantage_margin"] == 0.0
        assert configured["supervisor_gate"]["default_to_supervisor"] is True
        assert configured["supervisor_gate"]["min_train_steps_before_policy_gate"] == 0
        assert configured["agent"]["supervisor_gate"] == configured["supervisor_gate"]
        assert configured["agent"]["exploration_mode"] == "epsilon"
        assert configured["agent"]["eps_start"] == 0.2
        assert configured["agent"]["eps_end"] == 0.02
        assert configured["agent"]["eps_decay_steps"] == 38_000
        safety = configured["horizon_safety"]
        assert safety["enabled"] is False
        assert safety["release_filter"]["enabled"] is False
        assert safety["reward_probation"]["enabled"] is False
        assert safety["shadow_default_mpc"]["enabled"] is False

    assert standard["agent"]["multistep_mode"] == "one_step"
    assert dueling["agent"]["multistep_mode"] == "n_step"


def test_polymer_wrapper_modules_reference_sg_agent_classes():
    import RL_assisted_MPC_horizons_supervisor_gated_dqn_unified as standard
    import RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified as dueling

    assert standard.SupervisorGatedDQNAgent is SupervisorGatedDQNAgent
    assert dueling.SupervisorGatedDuelingDQNAgent is SupervisorGatedDuelingDQNAgent


def test_polymer_exported_scripts_expose_wrapper_hooks_without_execution():
    standard_source = (ROOT / "RL_assisted_MPC_horizons_unified.py").read_text(encoding="utf-8")
    dueling_source = (ROOT / "RL_assisted_MPC_horizons_dueling_unified.py").read_text(encoding="utf-8")

    for source in (standard_source, dueling_source):
        assert "NB_CONFIGURE" in source
        assert "NOTEBOOK_SOURCE_OVERRIDE" in source
        assert "RUN_SUMMARY_TITLE_OVERRIDE" in source
        assert "HORIZON_AGENT_CLASS_OVERRIDE" in source
        assert "supervisor_gate_config" in source


def test_sg_horizon_helper_tie_defaults_to_supervisor():
    agent = make_sg_dqn()
    set_dqn_q_values(agent, [0.0, 1.0, 0.5])
    decision = select_supervisor_gated_horizon_action(
        agent=agent,
        state=np.zeros(3, dtype=np.float32),
        step=4,
        warm_start_step=0,
        decision_interval=4,
        default_action=1,
        supervisor_action=1,
        last_action=None,
        test=False,
    )
    assert decision.action == 1
    assert decision.policy_action == 1
    assert decision.supervisor_action == 1
    assert decision.selected_source == SOURCE_SUPERVISOR


def test_sg_horizon_helper_selects_policy_when_q_advantage_is_positive():
    agent = make_sg_dqn()
    set_dqn_q_values(agent, [0.0, 1.0, 3.0])
    decision = select_supervisor_gated_horizon_action(
        agent=agent,
        state=np.zeros(3, dtype=np.float32),
        step=4,
        warm_start_step=0,
        decision_interval=4,
        default_action=0,
        supervisor_action=0,
        last_action=None,
        test=False,
    )
    assert decision.action == 2
    assert decision.policy_action == 2
    assert decision.supervisor_action == 0
    assert decision.selected_source == SOURCE_POLICY
    assert decision.advantage_policy_supervisor > 0.0


def test_held_interval_is_logged_as_held():
    agent = make_sg_dqn()
    set_dqn_q_values(agent, [0.0, 1.0, 3.0])
    decision = select_supervisor_gated_horizon_action(
        agent=agent,
        state=np.zeros(3, dtype=np.float32),
        step=5,
        warm_start_step=0,
        decision_interval=4,
        default_action=0,
        supervisor_action=0,
        last_action=2,
        test=False,
    )
    assert decision.action == 2
    assert decision.decision_taken == 0
    assert decision.source == ACTION_SOURCE_HELD_INTERVAL
    assert decision.selected_source == SOURCE_HELD


def test_supervised_replay_helper_pushes_final_executed_action():
    class RecorderAgent:
        def __init__(self):
            self.calls = []
            self.train_calls = 0

        def push_supervised(self, *args, **kwargs):
            self.calls.append((args, kwargs))

        def train_step(self):
            self.train_calls += 1
            return 1.0

    agent = RecorderAgent()
    decision = select_supervisor_gated_horizon_action(
        agent=make_sg_dqn(),
        state=np.zeros(3, dtype=np.float32),
        step=5,
        warm_start_step=0,
        decision_interval=4,
        default_action=0,
        supervisor_action=0,
        last_action=2,
        test=False,
    )
    result = replay_train_supervisor_gated_horizon_agent(
        agent=agent,
        state=np.zeros(3, dtype=np.float32),
        action=1,
        reward=2.0,
        next_state=np.ones(3, dtype=np.float32),
        done=0.0,
        step=5,
        test=False,
        replay_start_step=0,
        train_start_step=5,
        decision=decision,
    )
    assert result["pushed"] is True
    assert result["trained"] is True
    args, kwargs = agent.calls[0]
    assert int(args[1]) == 1
    assert int(kwargs["policy_action"]) == int(decision.policy_action)
    assert int(kwargs["supervisor_action"]) == int(decision.supervisor_action)
    assert int(kwargs["selected_source"]) == SOURCE_HELD
    assert agent.train_calls == 1


def run_direct():
    test_wrapper_configs_set_sg_defaults_and_disable_old_safety()
    test_distillation_dueling_aspen6_legacy_reward_wrapper_config()
    test_wrapper_modules_reference_sg_agent_classes()
    test_polymer_wrapper_configs_set_sg_defaults_and_disable_old_safety()
    test_polymer_wrapper_modules_reference_sg_agent_classes()
    test_polymer_exported_scripts_expose_wrapper_hooks_without_execution()
    test_sg_horizon_helper_tie_defaults_to_supervisor()
    test_sg_horizon_helper_selects_policy_when_q_advantage_is_positive()
    test_held_interval_is_logged_as_held()
    test_supervised_replay_helper_pushes_final_executed_action()
    print("supervisor_gated_horizon_runner tests passed")


if __name__ == "__main__":
    run_direct()
