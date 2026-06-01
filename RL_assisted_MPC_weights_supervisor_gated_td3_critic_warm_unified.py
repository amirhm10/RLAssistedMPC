"""Polymer SG-TD3 weight-multiplier runner with conservative critic-warm release.

This entrypoint is a clean ablation of the standard TD3 weights runner. It uses
the identity weight multiplier as the supervisor action and disables behavioral
cloning, handoff blending, authority ramps, reward probation, and shadow
diagnostics for this runner only.
"""

from __future__ import annotations

import runpy
from copy import deepcopy
from pathlib import Path


THIS_RUNNER = Path(__file__).name
BASE_WEIGHTS_RUNNER = Path(__file__).with_name("RL_assisted_MPC_weights_unified.py")


def configure_sg_td3_weights_critic_warm(nb: dict) -> dict:
    """Apply runner-local SG-TD3 critic-warm settings for polymer weights."""
    nb = deepcopy(nb)

    nb["agent_kind"] = "sg_td3"
    nb["run_mode"] = "disturb"
    nb["warm_start_override"] = 10
    nb["post_warm_start_action_freeze_subepisodes"] = 3
    nb["post_warm_start_actor_freeze_subepisodes"] = 3
    nb["result_prefix_override"] = "sg_td3_weights_critic_warm3_conservative_disturb"
    nb["compare_prefix_override"] = "disturb_compare_sg_td3_weights_critic_warm3_conservative"

    profiles = deepcopy(nb.get("run_profiles", {}))
    profiles[("sg_td3", "nominal")] = {
        "result_prefix": "sg_td3_weights_critic_warm3_conservative_nominal",
        "compare_prefix": "nominal_compare_sg_td3_weights_critic_warm3_conservative",
        "compare_mode": "nominal",
        "plot_start_episode": 2,
        "compare_start_episode": 2,
    }
    profiles[("sg_td3", "disturb")] = {
        "result_prefix": "sg_td3_weights_critic_warm3_conservative_disturb",
        "compare_prefix": "disturb_compare_sg_td3_weights_critic_warm3_conservative",
        "compare_mode": "disturb",
        "plot_start_episode": 2,
        "compare_start_episode": 2,
    }
    nb["run_profiles"] = profiles

    bc_cfg = deepcopy(nb.get("behavioral_cloning", {}))
    bc_cfg["enabled"] = False
    if not isinstance(bc_cfg.get("handoff"), dict):
        bc_cfg["handoff"] = {}
    bc_cfg["handoff"]["enabled"] = False
    bc_cfg["handoff"]["start_authority"] = 1.0
    bc_cfg["handoff"]["end_authority"] = 1.0
    bc_cfg["handoff"]["active_subepisodes"] = 0
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

    safety_cfg = deepcopy(nb.get("weight_safety", {}))
    safety_cfg["enabled"] = True
    safety_cfg["fallback_to_identity_on_nonfinite"] = True
    safety_cfg["fallback_to_identity_on_solve_failure"] = False
    safety_cfg.setdefault("reward_probation", {})
    safety_cfg["reward_probation"]["enabled"] = False
    safety_cfg.setdefault("shadow_identity_mpc", {})
    safety_cfg["shadow_identity_mpc"]["enabled"] = False
    nb["weight_safety"] = safety_cfg

    gate_cfg = deepcopy(nb.get("supervisor_gate", {}))
    gate_cfg["advantage_margin"] = 0.5
    gate_cfg["score_uncertainty_weight"] = 0.5
    gate_cfg["score_supervisor_action_weight"] = 0.05
    gate_cfg["score_previous_action_weight"] = 0.01
    gate_cfg["supervisor_bc_weight"] = 0.0
    gate_cfg["enable_supervisor_actor_loss"] = False
    gate_cfg["min_train_steps_before_policy_gate"] = 0
    nb["supervisor_gate"] = gate_cfg

    return nb


def main() -> dict:
    run_globals = runpy.run_path(
        str(BASE_WEIGHTS_RUNNER),
        run_name="__main__",
        init_globals={
            "NB_CONFIGURE": configure_sg_td3_weights_critic_warm,
            "NOTEBOOK_SOURCE_OVERRIDE": THIS_RUNNER,
            "RUN_SUMMARY_TITLE_OVERRIDE": "Polymer Weight SG-TD3 Conservative Critic-Warm-3 run summary",
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
