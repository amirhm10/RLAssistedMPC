from __future__ import annotations

import inspect
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL_assisted_MPC_markov_supervisor_gated_sac_critic_warm_unified import (
    configure_sg_sac_markov_critic_warm,
)
from RL_assisted_MPC_markov_supervisor_gated_sac_critic_warm_standard_standard_unified import (
    configure_sg_sac_markov_critic_warm_standard_standard,
)
from RL_assisted_MPC_residual_supervisor_gated_sac_critic_warm_unified import (
    configure_sg_sac_residual_critic_warm,
)
from RL_assisted_MPC_weights_supervisor_gated_sac_critic_warm_unified import (
    configure_sg_sac_weights_critic_warm,
)
from SACAgent.supervisor_gated_sac_agent import SupervisorGatedSACAgent
from systems.polymer import get_polymer_notebook_defaults
from utils.markov_runner import make_td3_markov_agent
from utils.residual_runner import run_residual_supervisor
from utils.weights_runner import run_weight_multiplier_supervisor


def _assert_common_sg_sac_config(configured):
    assert configured["agent_kind"] == "sg_sac"
    assert configured["run_mode"] == "disturb"
    assert configured["warm_start_override"] == 10
    assert configured["post_warm_start_action_freeze_subepisodes"] == 10
    assert configured["post_warm_start_actor_freeze_subepisodes"] == 3

    gate = configured["supervisor_gate"]
    assert gate["candidate_mode"] == "deterministic"
    assert gate["critic_dominance_gate_enabled"] is True
    assert gate["enable_supervisor_actor_loss"] is True
    assert gate["supervisor_bc_weight"] == 0.01
    assert gate["sampled_supervisor_bc_weight"] == 0.01
    assert gate["smooth_action_weight"] == 0.001
    assert gate["score_uncertainty_weight"] == 0.5
    assert gate["score_previous_action_weight"] == 0.01
    assert gate["min_train_steps_before_policy_gate"] == 0

    sac_cfg = configured["sac_agent"]
    assert sac_cfg["multistep_mode"] == "one_step"
    assert sac_cfg["n_step"] == 1
    assert sac_cfg["alpha_freeze"] == "actor_freeze"


def test_polymer_sg_sac_profiles_available():
    weights = get_polymer_notebook_defaults("weights")
    residual = get_polymer_notebook_defaults("residual")
    markov = get_polymer_notebook_defaults("markov")

    for nb in (weights, residual, markov):
        assert ("sg_sac", "nominal") in nb["run_profiles"]
        assert ("sg_sac", "disturb") in nb["run_profiles"]

    assert "sac_agent" in markov


def test_weights_sg_sac_wrapper_config_shadow_identity():
    configured = configure_sg_sac_weights_critic_warm(get_polymer_notebook_defaults("weights"))
    _assert_common_sg_sac_config(configured)
    assert configured["state_mode"] == "standard"
    assert configured["supervisor_gate"]["advantage_margin"] == 0.5
    assert configured["supervisor_gate"]["critic_dominance_margin"] == 0.5
    assert configured["supervisor_gate"]["score_supervisor_action_weight"] == 0.05
    assert "detgate_hidden7" in configured["result_prefix_override"]
    assert "detgate_hidden7" in configured["compare_prefix_override"]
    assert "standard" in configured["result_prefix_override"]
    assert "standard" in configured["compare_prefix_override"]

    bc = configured["behavioral_cloning"]
    assert bc["enabled"] is False
    assert bc["handoff"]["enabled"] is False
    assert bc["release_gate"]["enabled"] is False

    safety = configured["weight_safety"]
    assert safety["enabled"] is True
    assert safety["fallback_to_identity_on_nonfinite"] is True
    assert safety["fallback_to_identity_on_solve_failure"] is False
    assert safety["reward_probation"]["enabled"] is False
    assert safety["shadow_identity_mpc"]["enabled"] is True


def test_residual_sg_sac_wrapper_config_shadow_only():
    configured = configure_sg_sac_residual_critic_warm(get_polymer_notebook_defaults("residual"))
    _assert_common_sg_sac_config(configured)
    assert configured["state_mode"] == "standard"
    assert configured["supervisor_gate"]["advantage_margin"] == 0.5
    assert configured["supervisor_gate"]["critic_dominance_margin"] == 0.5
    assert configured["supervisor_gate"]["score_supervisor_action_weight"] == 0.05
    assert "detgate_hidden7" in configured["result_prefix_override"]
    assert "detgate_hidden7" in configured["compare_prefix_override"]
    assert "standard" in configured["result_prefix_override"]
    assert "standard" in configured["compare_prefix_override"]

    assert configured["residual_authority_enabled"] is False
    assert configured["authority_use_rho"] is False
    assert configured["use_rho_authority"] is False
    assert configured["append_rho_to_state"] is False
    assert configured["residual_zero_deadband_enabled"] is False

    bc = configured["behavioral_cloning"]
    assert bc["enabled"] is False
    assert bc["handoff"]["enabled"] is False
    assert bc["release_gate"]["enabled"] is False

    safety = configured["residual_safety"]
    assert safety["enabled"] is True
    assert safety["fallback_to_zero_on_nonfinite"] is True
    assert safety["reward_probation"]["enabled"] is False
    assert safety["early_release_guard"]["enabled"] is False
    assert safety["shadow_rho_authority"]["enabled"] is True
    assert safety["shadow_residual_deadband"]["enabled"] is True
    assert safety["shadow_direction_risk"]["enabled"] is True


def test_markov_sg_sac_wrapper_config_shadow_only():
    configured = configure_sg_sac_markov_critic_warm(get_polymer_notebook_defaults("markov"))
    _assert_common_sg_sac_config(configured)
    assert configured["state_mode"] == "standard"
    assert configured["markov_supervisor_mode"] == "ls_else_mpc"
    assert configured["markov_live_safety_mode"] == "shadow_only"
    assert configured["supervisor_gate"]["advantage_margin"] == 0.0
    assert configured["supervisor_gate"]["critic_dominance_margin"] == 0.0
    assert configured["supervisor_gate"]["score_supervisor_action_weight"] == 0.02
    assert "detgate_hidden7" in configured["result_prefix_override"]
    assert "detgate_hidden7" in configured["compare_prefix_override"]
    assert "standard" in configured["result_prefix_override"]
    assert "standard" in configured["compare_prefix_override"]

    bc = configured["behavioral_cloning"]
    assert bc["enabled"] is False
    assert bc["handoff"]["enabled"] is False
    assert bc["release_gate"]["enabled"] is False

    ctrl = configured["controller"]
    assert ctrl["z_safety"]["enabled"] is False
    assert ctrl["td3_priority_fallback"]["enabled"] is False
    assert ctrl["td3_authority_ramp"]["enabled"] is False
    assert ctrl["rl_fallback_to_ls"] is False
    assert ctrl["force_td3_execute"] is False
    assert ctrl["force_td3_respects_warm_start"] is True
    assert ctrl["markov_shadow_safety"]["enabled"] is True
    assert ctrl["markov_shadow_safety"]["compute_ls_candidate"] is False


def test_markov_sg_sac_standard_standard_wrapper_config_base_only():
    configured = configure_sg_sac_markov_critic_warm_standard_standard(get_polymer_notebook_defaults("markov"))
    _assert_common_sg_sac_config(configured)
    assert configured["state_mode"] == "standard"
    assert configured["markov_agent_state_features"] == "base_only"
    assert configured["markov_supervisor_mode"] == "ls_else_mpc"
    assert configured["markov_live_safety_mode"] == "shadow_only"
    assert configured["supervisor_gate"]["candidate_mode"] == "deterministic"
    assert configured["supervisor_gate"]["critic_dominance_margin"] == 0.0
    assert "standard_standard" in configured["result_prefix_override"]
    assert "standard_standard" in configured["compare_prefix_override"]
    assert configured["run_profiles"][("sg_sac", "disturb")]["result_prefix"].endswith("standard_standard")


def test_markov_sg_sac_construction_returns_supervisor_gated_sac_agent():
    configured = configure_sg_sac_markov_critic_warm(get_polymer_notebook_defaults("markov"))
    configured["sac_agent"].update(
        {
            "actor_hidden": [],
            "critic_hidden": [],
            "batch_size": 4,
            "buffer_size": 32,
            "replay_recent_window": 8,
        }
    )
    agent = make_td3_markov_agent(configured, state_dim=12, action_dim=4, set_points_len=20)
    assert isinstance(agent, SupervisorGatedSACAgent)
    assert agent.multistep_mode == "one_step"
    assert agent.supervisor_gate_config.candidate_mode == "deterministic"
    assert agent.supervisor_gate_config.critic_dominance_gate_enabled is True
    assert agent.supervisor_gate_config.supervisor_bc_weight == 0.01
    assert agent.supervisor_gate_config.sampled_supervisor_bc_weight == 0.01
    assert agent.supervisor_gate_config.smooth_action_weight == 0.001


def test_shared_continuous_runners_accept_sg_sac_without_closed_loop_execution():
    weight_source = inspect.getsource(run_weight_multiplier_supervisor)
    residual_source = inspect.getsource(run_residual_supervisor)
    assert '"sg_sac"' in weight_source
    assert '"sg_sac"' in residual_source


def run_direct():
    test_polymer_sg_sac_profiles_available()
    test_weights_sg_sac_wrapper_config_shadow_identity()
    test_residual_sg_sac_wrapper_config_shadow_only()
    test_markov_sg_sac_wrapper_config_shadow_only()
    test_markov_sg_sac_standard_standard_wrapper_config_base_only()
    test_markov_sg_sac_construction_returns_supervisor_gated_sac_agent()
    test_shared_continuous_runners_accept_sg_sac_without_closed_loop_execution()
    print("supervisor_gated_sac_runner tests passed")


if __name__ == "__main__":
    run_direct()
