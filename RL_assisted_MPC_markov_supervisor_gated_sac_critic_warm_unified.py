"""Polymer SG-SAC Markov runner with deterministic LS-or-MPC supervisor gate.

This entrypoint mirrors the SG-TD3 Markov critic-warm runner while using SAC as
the policy. Live Markov safety layers are disabled; their previous settings are
kept as shadow diagnostics in the result bundle. The first 3 hidden
subepisodes are critic-only, followed by 7 hidden subepisodes where actor/alpha
train while the LS-or-MPC supervisor still executes.
"""

from __future__ import annotations

import runpy
from copy import deepcopy
from pathlib import Path


THIS_RUNNER = Path(__file__).name
BASE_MARKOV_RUNNER = Path(__file__).with_name("RL_assisted_MPC_markov_unified.py")


def _force_sac_one_step(nb: dict) -> None:
    sac_cfg = deepcopy(nb.get("sac_agent", {}))
    sac_cfg["n_step"] = 1
    sac_cfg["multistep_mode"] = "one_step"
    sac_cfg["alpha_freeze"] = "actor_freeze"
    sac_cfg["actor_q_mode"] = "min"
    nb["sac_agent"] = sac_cfg


def configure_sg_sac_markov_critic_warm(nb: dict) -> dict:
    """Apply runner-local SG-SAC critic-warm settings for polymer Markov."""
    nb = deepcopy(nb)

    active_controller_defaults = deepcopy(nb.get("controller", {}))
    active_bc_defaults = deepcopy(nb.get("behavioral_cloning", {}))

    nb["agent_kind"] = "sg_sac"
    nb["run_mode"] = "disturb"
    nb["state_mode"] = "standard"
    nb["warm_start_override"] = 10
    nb["post_warm_start_action_freeze_subepisodes"] = 10
    nb["post_warm_start_actor_freeze_subepisodes"] = 3
    nb["markov_supervisor_mode"] = "ls_else_mpc"
    nb["markov_live_safety_mode"] = "shadow_only"
    nb["result_prefix_override"] = "sg_sac_markov_detgate_hidden7_ls_else_mpc_shadow_disturb_standard"
    nb["compare_prefix_override"] = "disturb_compare_sg_sac_markov_detgate_hidden7_ls_else_mpc_shadow_standard"

    profiles = deepcopy(nb.get("run_profiles", {}))
    profiles[("sg_sac", "nominal")] = {
        "result_prefix": "sg_sac_markov_detgate_hidden7_ls_else_mpc_shadow_nominal_standard",
        "compare_prefix": "nominal_compare_sg_sac_markov_detgate_hidden7_ls_else_mpc_shadow_standard",
        "compare_mode": "nominal",
        "plot_start_episode": 2,
        "compare_start_episode": 2,
    }
    profiles[("sg_sac", "disturb")] = {
        "result_prefix": "sg_sac_markov_detgate_hidden7_ls_else_mpc_shadow_disturb_standard",
        "compare_prefix": "disturb_compare_sg_sac_markov_detgate_hidden7_ls_else_mpc_shadow_standard",
        "compare_mode": "disturb",
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
    ctrl["z_safety"] = {"enabled": False}
    ctrl["td3_priority_fallback"] = {"enabled": False}
    ctrl["td3_authority_ramp"] = {"enabled": False}
    ctrl["markov_shadow_safety"] = {
        "enabled": True,
        "compute_ls_candidate": False,
        "z_safety": active_controller_defaults.get("z_safety", {}),
        "td3_priority_fallback": active_controller_defaults.get("td3_priority_fallback", {}),
        "bc_handoff": active_bc_defaults.get("handoff") if isinstance(active_bc_defaults.get("handoff"), dict) else {},
    }
    ctrl["rl_store_executed_action_in_replay"] = True
    nb["controller"] = ctrl

    gate_cfg = deepcopy(nb.get("supervisor_gate", {}))
    gate_cfg["candidate_mode"] = "deterministic"
    gate_cfg["advantage_margin"] = 0.0
    gate_cfg["critic_dominance_gate_enabled"] = True
    gate_cfg["critic_dominance_margin"] = 0.0
    gate_cfg["score_uncertainty_weight"] = 0.5
    gate_cfg["score_supervisor_action_weight"] = 0.02
    gate_cfg["score_previous_action_weight"] = 0.01
    gate_cfg["enable_supervisor_actor_loss"] = True
    gate_cfg["supervisor_bc_weight"] = 0.01
    gate_cfg["sampled_supervisor_bc_weight"] = 0.01
    gate_cfg["smooth_action_weight"] = 0.001
    gate_cfg["min_train_steps_before_policy_gate"] = 0
    nb["supervisor_gate"] = gate_cfg

    _force_sac_one_step(nb)
    return nb


def main() -> dict:
    run_globals = runpy.run_path(
        str(BASE_MARKOV_RUNNER),
        run_name="__main__",
        init_globals={
            "NB_CONFIGURE": configure_sg_sac_markov_critic_warm,
            "NOTEBOOK_SOURCE_OVERRIDE": THIS_RUNNER,
            "RUN_SUMMARY_TITLE_OVERRIDE": "Polymer Markov SG-SAC Critic-Warm-3 LS-or-MPC run summary",
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
