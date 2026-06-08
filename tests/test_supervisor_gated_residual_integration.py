from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_polymer_weights_and_residual_sg_td3_defaults():
    from systems.polymer import get_polymer_notebook_defaults

    for family in ("weights", "residual"):
        nb = get_polymer_notebook_defaults(family)
        assert nb["agent_kind"] == "sg_td3"
        assert nb["run_mode"] == "disturb"
        assert nb["state_mode"] == "mismatch"
        assert ("td3", "disturb") in nb["run_profiles"]
        assert ("sg_td3", "disturb") in nb["run_profiles"]
        assert ("sg_sac", "disturb") not in nb["run_profiles"]
        assert ("td7", "disturb") not in nb["run_profiles"]
        assert nb["episode_defaults"]["warm_start"] == 10
        assert nb["post_warm_start_action_freeze_subepisodes"] == 3
        assert nb["post_warm_start_actor_freeze_subepisodes"] == 3
        gate_cfg = nb["supervisor_gate"]
        assert gate_cfg["advantage_margin"] == 0.5
        assert gate_cfg["score_uncertainty_weight"] == 0.5
        assert gate_cfg["score_supervisor_action_weight"] == 0.05
        assert gate_cfg["score_previous_action_weight"] == 0.01
        assert gate_cfg["supervisor_bc_weight"] == 0.0
        assert gate_cfg["enable_supervisor_actor_loss"] is False
        assert gate_cfg["min_train_steps_before_policy_gate"] == 0

    weights = get_polymer_notebook_defaults("weights")
    assert weights["behavioral_cloning"]["enabled"] is False
    assert weights["behavioral_cloning"]["handoff"]["enabled"] is False
    assert weights["weight_safety"]["enabled"] is True
    assert weights["weight_safety"]["fallback_to_identity_on_nonfinite"] is True
    assert weights["weight_safety"]["fallback_to_identity_on_solve_failure"] is False

    residual = get_polymer_notebook_defaults("residual")
    assert residual["residual_authority_enabled"] is False
    assert residual["authority_use_rho"] is False
    assert residual["append_rho_to_state"] is False
    assert residual["residual_zero_deadband_enabled"] is False
    assert residual["behavioral_cloning"]["enabled"] is False
    assert residual["behavioral_cloning"]["handoff"]["enabled"] is False
    assert residual["td3_authority_ramp"]["enabled"] is False
    assert residual["residual_safety"]["fallback_to_zero_on_nonfinite"] is True
    assert residual["residual_safety"]["early_release_guard"]["enabled"] is False
    assert residual["residual_safety"]["shadow_rho_authority"]["enabled"] is False
    assert residual["residual_safety"]["shadow_residual_deadband"]["enabled"] is False
    assert residual["residual_safety"]["shadow_direction_risk"]["enabled"] is False


def test_distillation_residual_sg_td3_profile_available():
    from systems.distillation import get_distillation_notebook_defaults

    nb = get_distillation_notebook_defaults("residual")
    assert ("sg_td3", "nominal", "none") in nb["run_profiles"]
    assert ("sg_td3", "disturb", "ramp") in nb["run_profiles"]
    assert ("sg_td3", "disturb", "fluctuation") in nb["run_profiles"]
    assert "supervisor_gate" in nb


def test_distillation_sg_td3_critic_warm_config_manual_layers_off():
    from distillation_RL_assisted_MPC_residual_supervisor_gated_td3_critic_warm_unified import (
        configure_sg_td3_residual_critic_warm,
    )
    from systems.distillation import get_distillation_notebook_defaults

    configured = configure_sg_td3_residual_critic_warm(get_distillation_notebook_defaults("residual"))
    assert configured["agent_kind"] == "sg_td3"
    assert configured["run_mode"] == "disturb"
    assert configured["disturbance_profile"] == "fluctuation"
    assert configured["state_mode"] == "mismatch"
    assert "mismatch" in configured["result_prefix_override"]
    assert "mismatch" in configured["compare_prefix_override"]
    assert "critic_warm3" in configured["result_prefix_override"]
    assert "critic_warm3" in configured["compare_prefix_override"]
    assert "margin05" in configured["result_prefix_override"]
    assert "margin05" in configured["compare_prefix_override"]
    assert "paramnoise" in configured["result_prefix_override"]
    assert "paramnoise" in configured["compare_prefix_override"]
    assert configured["warm_start_override"] == 10
    assert configured["post_warm_start_action_freeze_subepisodes"] == 3
    assert configured["post_warm_start_actor_freeze_subepisodes"] == 3
    assert configured["residual_authority_enabled"] is False
    assert configured["authority_use_rho"] is False
    assert configured["use_rho_authority"] is False
    assert configured["append_rho_to_state"] is False
    assert configured["residual_zero_deadband_enabled"] is False

    bc_cfg = configured["behavioral_cloning"]
    assert bc_cfg["enabled"] is False
    assert bc_cfg["handoff"]["enabled"] is False
    assert bc_cfg["release_gate"]["enabled"] is False

    ramp_cfg = configured["td3_authority_ramp"]
    assert ramp_cfg["enabled"] is False
    assert ramp_cfg["diagnostic_release_gate_only"] is False

    td3_cfg = configured["td3_agent"]
    assert td3_cfg["exploration_mode"] == "param_noise"
    assert td3_cfg["param_noise_std_start"] == 0.10
    assert td3_cfg["param_noise_std_end"] == 0.02
    assert td3_cfg["param_noise_resample_interval"] == 4

    safety_cfg = configured["residual_safety"]
    assert safety_cfg["enabled"] is True
    assert safety_cfg["fallback_to_zero_on_nonfinite"] is True
    assert safety_cfg["reward_probation"]["enabled"] is False
    assert safety_cfg["early_release_guard"]["enabled"] is False
    assert safety_cfg["shadow_rho_authority"]["enabled"] is False
    assert safety_cfg["shadow_residual_deadband"]["enabled"] is False
    assert safety_cfg["shadow_direction_risk"]["enabled"] is False

    gate_cfg = configured["supervisor_gate"]
    assert gate_cfg["advantage_margin"] == 0.5
    assert gate_cfg["score_uncertainty_weight"] == 0.5
    assert gate_cfg["score_supervisor_action_weight"] == 0.05
    assert gate_cfg["score_previous_action_weight"] == 0.01
    assert gate_cfg["supervisor_bc_weight"] == 0.0
    assert gate_cfg["enable_supervisor_actor_loss"] is False
    assert gate_cfg["min_train_steps_before_policy_gate"] == 0


def test_distillation_sg_td3_weights_and_markov_use_mismatch_state():
    from distillation_RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_unified import (
        configure_sg_td3_markov_critic_warm,
    )
    from distillation_RL_assisted_MPC_weights_supervisor_gated_td3_critic_warm_unified import (
        configure_sg_td3_weights_critic_warm,
    )
    from systems.distillation import get_distillation_notebook_defaults

    weights = configure_sg_td3_weights_critic_warm(get_distillation_notebook_defaults("weights"))
    markov = configure_sg_td3_markov_critic_warm(get_distillation_notebook_defaults("markov"))

    for configured in (weights, markov):
        assert configured["agent_kind"] == "sg_td3"
        assert configured["run_mode"] == "disturb"
        assert configured["disturbance_profile"] == "fluctuation"
        assert configured["state_mode"] == "mismatch"
        assert "mismatch" in configured["result_prefix_override"]
        assert "mismatch" in configured["compare_prefix_override"]

    assert "critic_warm3" in weights["result_prefix_override"]
    assert "critic_warm3" in weights["compare_prefix_override"]
    assert weights["post_warm_start_action_freeze_subepisodes"] == 3
    assert weights["post_warm_start_actor_freeze_subepisodes"] == 3

    assert "critic_warm3" in markov["result_prefix_override"]
    assert "critic_warm3" in markov["compare_prefix_override"]
    assert "margin05" in markov["result_prefix_override"]
    assert "margin05" in markov["compare_prefix_override"]
    assert "softparamnoise" in markov["result_prefix_override"]
    assert "softparamnoise" in markov["compare_prefix_override"]
    assert markov["post_warm_start_action_freeze_subepisodes"] == 3
    assert markov["post_warm_start_actor_freeze_subepisodes"] == 3

    assert markov["markov_supervisor_mode"] == "ls_else_mpc"
    assert markov["markov_live_safety_mode"] == "shadow_only"
    assert markov["td3_agent"]["exploration_mode"] == "param_noise"
    assert markov["td3_agent"]["param_noise_std_start"] == 0.10
    assert markov["td3_agent"]["param_noise_std_end"] == 0.02
    assert markov["td3_agent"]["param_noise_resample_interval"] == 4
    assert markov["supervisor_gate"]["advantage_margin"] == 0.5


def test_polymer_weights_and_residual_simple_runners_are_td3_only():
    weights_source = (ROOT / "RL_assisted_MPC_weights_unified.py").read_text(encoding="utf-8")
    residual_source = (ROOT / "RL_assisted_MPC_residual_unified.py").read_text(encoding="utf-8")

    for source in (weights_source, residual_source):
        assert "SupervisorGatedTD3Agent" in source
        assert "AGENT_KIND not in" in source
        assert "sg_td3" in source
        assert "sg_sac" not in source
        assert "SupervisorGatedSACAgent" not in source


def test_residual_runner_imports_with_supervisor_gated_branch():
    from utils.residual_runner import run_residual_supervisor

    assert callable(run_residual_supervisor)


def run_direct():
    test_polymer_weights_and_residual_sg_td3_defaults()
    test_distillation_residual_sg_td3_profile_available()
    test_distillation_sg_td3_critic_warm_config_manual_layers_off()
    test_distillation_sg_td3_weights_and_markov_use_mismatch_state()
    test_polymer_weights_and_residual_simple_runners_are_td3_only()
    test_residual_runner_imports_with_supervisor_gated_branch()
    print("supervisor_gated_residual_integration tests passed")


if __name__ == "__main__":
    run_direct()
