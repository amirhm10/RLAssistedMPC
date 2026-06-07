"""Distillation SG-TD3 Markov runner with LS-or-MPC supervisor and critic-warm release.

This entrypoint mirrors the polymer Markov SG-TD3 critic-warm ablation for the
Aspen C2 splitter case. It executes the dynamic LS-or-MPC Markov supervisor
during warm start and a 3-subepisode critic-only window, then lets SG-TD3 choose
between the actor and the same supervisor action. Live Markov safety layers are
disabled, while their shadow diagnostics remain logged in the result bundle.
"""

from __future__ import annotations

import runpy
from copy import deepcopy
from pathlib import Path


THIS_RUNNER = Path(__file__).name
BASE_MARKOV_RUNNER = Path(__file__).with_name("distillation_RL_assisted_MPC_markov_unified.py")


def configure_sg_td3_markov_critic_warm(nb: dict) -> dict:
    """Apply runner-local SG-TD3 critic-warm settings for distillation Markov."""
    nb = deepcopy(nb)

    active_controller_defaults = deepcopy(nb.get("controller", {}))
    active_bc_defaults = deepcopy(nb.get("behavioral_cloning", {}))

    nb["agent_kind"] = "sg_td3"
    nb["run_mode"] = "disturb"
    nb["disturbance_profile"] = "fluctuation"
    nb["state_mode"] = "mismatch"
    nb["warm_start_override"] = 10
    nb["post_warm_start_action_freeze_subepisodes"] = 3
    nb["post_warm_start_actor_freeze_subepisodes"] = 3
    nb["markov_supervisor_mode"] = "ls_else_mpc"
    nb["markov_live_safety_mode"] = "shadow_only"
    nb["result_prefix_override"] = (
        "distillation_markov_sg_td3_critic_warm3_margin05_softparamnoise_ls_else_mpc_shadow_disturb_fluctuation_mismatch"
    )
    nb["compare_prefix_override"] = (
        "distillation_compare_markov_sg_td3_critic_warm3_margin05_softparamnoise_ls_else_mpc_shadow_disturb_fluctuation_mismatch"
    )

    profiles = deepcopy(nb.get("run_profiles", {}))
    profiles[("sg_td3", "nominal", "none")] = {
        "n_tests": 200,
        "set_points_len": 200,
        "warm_start": 10,
        "test_cycle": [False, False, False, False, False],
        "plot_start_episode": 2,
        "compare_start_episode": 2,
    }
    profiles[("sg_td3", "disturb", "ramp")] = {
        "n_tests": 200,
        "set_points_len": 200,
        "warm_start": 10,
        "test_cycle": [False, False, False, False, False],
        "plot_start_episode": 2,
        "compare_start_episode": 2,
    }
    profiles[("sg_td3", "disturb", "fluctuation")] = {
        "n_tests": 200,
        "set_points_len": 200,
        "warm_start": 10,
        "test_cycle": [False, False, False, False, False],
        "plot_start_episode": 2,
        "compare_start_episode": 2,
    }
    nb["run_profiles"] = profiles

    bc_cfg = deepcopy(nb.get("behavioral_cloning", {}))
    bc_cfg["enabled"] = False
    if not isinstance(bc_cfg.get("release_gate"), dict):
        bc_cfg["release_gate"] = {}
    bc_cfg["release_gate"]["enabled"] = False
    bc_cfg["release_gate"]["diagnostic_only"] = False
    if not isinstance(bc_cfg.get("handoff"), dict):
        bc_cfg["handoff"] = {}
    bc_cfg["handoff"]["enabled"] = False
    bc_cfg["handoff"]["start_authority"] = 1.0
    bc_cfg["handoff"]["end_authority"] = 1.0
    bc_cfg["handoff"]["active_subepisodes"] = 0
    if not isinstance(bc_cfg.get("tail_anchor"), dict):
        bc_cfg["tail_anchor"] = {}
    bc_cfg["tail_anchor"]["enabled"] = False
    nb["behavioral_cloning"] = bc_cfg

    ctrl = deepcopy(nb.get("controller", {}))
    ctrl["run_adaptive_ls"] = True
    ctrl["run_live_corrected_mpc"] = True
    ctrl["run_rl_proposal"] = True
    ctrl["rl_fallback_to_ls"] = False
    ctrl["force_td3_execute"] = False
    ctrl["force_td3_respects_warm_start"] = True
    ctrl["markov_supervisor_mode"] = "ls_else_mpc"
    ctrl["markov_live_safety_mode"] = "shadow_only"
    ctrl["z_bound"] = 0.05
    ctrl["z_safety"] = {"enabled": False}
    ctrl["td3_priority_fallback"] = {"enabled": False}
    ctrl["td3_authority_ramp"] = {"enabled": False}
    ctrl["markov_shadow_safety"] = {
        "enabled": True,
        "compute_ls_candidate": False,
        "z_safety": active_controller_defaults.get("z_safety", {}),
        "td3_priority_fallback": active_controller_defaults.get("td3_priority_fallback", {}),
        "bc_handoff": active_bc_defaults.get("handoff", {}),
    }
    ctrl["rl_store_executed_action_in_replay"] = True
    nb["controller"] = ctrl

    td3_cfg = deepcopy(nb.get("td3_agent", {}))
    td3_cfg["exploration_mode"] = "param_noise"
    td3_cfg["param_noise_std_start"] = 0.10
    td3_cfg["param_noise_std_end"] = 0.02
    td3_cfg["param_noise_resample_interval"] = 4
    nb["td3_agent"] = td3_cfg

    gate_cfg = deepcopy(nb.get("supervisor_gate", {}))
    gate_cfg["advantage_margin"] = 0.5
    gate_cfg["score_uncertainty_weight"] = 0.5
    gate_cfg["score_supervisor_action_weight"] = 0.02
    gate_cfg["score_previous_action_weight"] = 0.01
    gate_cfg["supervisor_bc_weight"] = 0.0
    gate_cfg["enable_supervisor_actor_loss"] = False
    gate_cfg["min_train_steps_before_policy_gate"] = 0
    nb["supervisor_gate"] = gate_cfg

    return nb


def main() -> dict:
    run_globals = runpy.run_path(
        str(BASE_MARKOV_RUNNER),
        run_name="__main__",
        init_globals={
            "NB_CONFIGURE": configure_sg_td3_markov_critic_warm,
            "NOTEBOOK_SOURCE_OVERRIDE": THIS_RUNNER,
            "RUN_SUMMARY_TITLE_OVERRIDE": "Distillation Markov SG-TD3 Critic-Warm-3 Margin-0.5 Soft-Param-Noise LS-or-MPC run summary",
        },
    )
    globals().update(
        {
            name: run_globals[name]
            for name in ("result_bundle", "out_dir_rl", "out_dir_cmp")
            if name in run_globals
        }
    )
    return run_globals


if __name__ == "__main__":
    main()
