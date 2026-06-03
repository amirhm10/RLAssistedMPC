"""Polymer SG-SAC residual runner with zero-residual supervisor.

This runner mirrors the SG-TD3 residual critic-warm experiment while using the
SupervisorGatedSACAgent. The live residual safety layers remain off so the
experiment isolates the SG candidate-vs-supervisor gate; shadow diagnostics
remain enabled in the saved bundle.
"""

from __future__ import annotations

import runpy
from copy import deepcopy
from pathlib import Path


THIS_RUNNER = Path(__file__).name
BASE_RESIDUAL_RUNNER = Path(__file__).with_name("RL_assisted_MPC_residual_unified.py")


def _force_sac_one_step(nb: dict) -> None:
    sac_cfg = deepcopy(nb.get("sac_agent", {}))
    sac_cfg["n_step"] = 1
    sac_cfg["multistep_mode"] = "one_step"
    sac_cfg["alpha_freeze"] = "actor_freeze"
    sac_cfg["actor_q_mode"] = "min"
    nb["sac_agent"] = sac_cfg


def configure_sg_sac_residual_critic_warm(nb: dict) -> dict:
    """Apply runner-local SG-SAC critic-warm settings for polymer residuals."""
    nb = deepcopy(nb)

    nb["agent_kind"] = "sg_sac"
    nb["run_mode"] = "disturb"
    nb["state_mode"] = "mismatch"
    nb["warm_start_override"] = 10
    nb["post_warm_start_action_freeze_subepisodes"] = 3
    nb["post_warm_start_actor_freeze_subepisodes"] = 3
    nb["result_prefix_override"] = "sg_sac_residual_critic_warm3_zero_shadow_disturb"
    nb["compare_prefix_override"] = "disturb_compare_sg_sac_residual_critic_warm3_zero_shadow"

    profiles = deepcopy(nb.get("run_profiles", {}))
    profiles[("sg_sac", "nominal")] = {
        "result_prefix": "sg_sac_residual_critic_warm3_zero_shadow_nominal",
        "compare_prefix": "nominal_compare_sg_sac_residual_critic_warm3_zero_shadow",
        "compare_mode": "nominal",
        "plot_start_episode": 2,
        "compare_start_episode": 2,
    }
    profiles[("sg_sac", "disturb")] = {
        "result_prefix": "sg_sac_residual_critic_warm3_zero_shadow_disturb",
        "compare_prefix": "disturb_compare_sg_sac_residual_critic_warm3_zero_shadow",
        "compare_mode": "disturb",
        "plot_start_episode": 2,
        "compare_start_episode": 2,
    }
    nb["run_profiles"] = profiles

    nb["residual_authority_enabled"] = False
    nb["authority_use_rho"] = False
    nb["use_rho_authority"] = False
    nb["append_rho_to_state"] = False
    nb["residual_zero_deadband_enabled"] = False

    bc_cfg = deepcopy(nb.get("behavioral_cloning", {}))
    bc_cfg["enabled"] = False
    if not isinstance(bc_cfg.get("handoff"), dict):
        bc_cfg["handoff"] = {}
    bc_cfg["handoff"]["enabled"] = False
    if not isinstance(bc_cfg.get("release_gate"), dict):
        bc_cfg["release_gate"] = {}
    bc_cfg["release_gate"]["enabled"] = False
    bc_cfg["release_gate"]["diagnostic_only"] = False
    if not isinstance(bc_cfg.get("tail_anchor"), dict):
        bc_cfg["tail_anchor"] = {}
    bc_cfg["tail_anchor"]["enabled"] = False
    nb["behavioral_cloning"] = bc_cfg

    ramp_cfg = deepcopy(nb.get("td3_authority_ramp", {}))
    ramp_cfg["enabled"] = False
    ramp_cfg["diagnostic_release_gate_only"] = False
    nb["td3_authority_ramp"] = ramp_cfg

    safety_cfg = deepcopy(nb.get("residual_safety", {}))
    safety_cfg["enabled"] = True
    safety_cfg.setdefault("reward_probation", {})
    safety_cfg["reward_probation"]["enabled"] = False
    safety_cfg["fallback_to_zero_on_nonfinite"] = True
    for key in ("shadow_rho_authority", "shadow_residual_deadband", "shadow_direction_risk"):
        safety_cfg.setdefault(key, {})
        safety_cfg[key]["enabled"] = True
    safety_cfg.setdefault("early_release_guard", {})
    safety_cfg["early_release_guard"]["enabled"] = False
    nb["residual_safety"] = safety_cfg

    gate_cfg = deepcopy(nb.get("supervisor_gate", {}))
    gate_cfg["advantage_margin"] = 0.5
    gate_cfg["score_uncertainty_weight"] = 0.5
    gate_cfg["score_supervisor_action_weight"] = 0.05
    gate_cfg["score_previous_action_weight"] = 0.01
    gate_cfg["enable_supervisor_actor_loss"] = True
    gate_cfg["supervisor_bc_weight"] = 0.01
    gate_cfg["smooth_action_weight"] = 0.001
    gate_cfg["min_train_steps_before_policy_gate"] = 0
    nb["supervisor_gate"] = gate_cfg

    _force_sac_one_step(nb)
    return nb


def main() -> dict:
    run_globals = runpy.run_path(
        str(BASE_RESIDUAL_RUNNER),
        run_name="__main__",
        init_globals={
            "NB_CONFIGURE": configure_sg_sac_residual_critic_warm,
            "NOTEBOOK_SOURCE_OVERRIDE": THIS_RUNNER,
            "RUN_SUMMARY_TITLE_OVERRIDE": "Polymer Residual SG-SAC Critic-Warm-3 run summary",
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
