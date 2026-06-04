"""Distillation SG-SAC weight-multiplier runner with deterministic hidden release.

This entrypoint transfers the polymer SG-SAC protected handover to the Aspen
C2 splitter weight supervisor. The identity weight multiplier is the
supervisor action, live manual safety and BC handoff layers are disabled, and
SG-SAC's deterministic twin-critic gate controls release. The first 3 hidden
subepisodes are critic-only, followed by 7 hidden subepisodes where actor/alpha
train while the identity supervisor still executes.
"""

from __future__ import annotations

import runpy
from copy import deepcopy
from pathlib import Path


THIS_RUNNER = Path(__file__).name
BASE_WEIGHTS_RUNNER = Path(__file__).with_name("distillation_RL_assisted_MPC_weights_unified.py")


def _force_sac_one_step(nb: dict) -> None:
    sac_cfg = deepcopy(nb.get("sac_agent", {}))
    sac_cfg["n_step"] = 1
    sac_cfg["multistep_mode"] = "one_step"
    sac_cfg["alpha_freeze"] = "actor_freeze"
    sac_cfg["actor_q_mode"] = "min"
    nb["sac_agent"] = sac_cfg


def configure_sg_sac_weights_critic_warm(nb: dict) -> dict:
    """Apply runner-local SG-SAC critic-warm settings for distillation weights."""
    nb = deepcopy(nb)

    nb["agent_kind"] = "sg_sac"
    nb["run_mode"] = "disturb"
    nb["disturbance_profile"] = "fluctuation"
    nb["state_mode"] = "mismatch"
    nb["warm_start_override"] = 10
    nb["post_warm_start_action_freeze_subepisodes"] = 10
    nb["post_warm_start_actor_freeze_subepisodes"] = 3
    nb["result_prefix_override"] = (
        "distillation_weights_sg_sac_detgate_hidden7_identity_shadow_disturb_fluctuation_mismatch"
    )
    nb["compare_prefix_override"] = (
        "distillation_compare_weights_sg_sac_detgate_hidden7_identity_shadow_disturb_fluctuation"
    )

    profiles = deepcopy(nb.get("run_profiles", {}))
    for profile_key in (
        ("sg_sac", "nominal", "none"),
        ("sg_sac", "disturb", "ramp"),
        ("sg_sac", "disturb", "fluctuation"),
    ):
        profiles[profile_key] = {
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
    safety_cfg["shadow_identity_mpc"]["enabled"] = True
    safety_cfg["shadow_identity_mpc"].setdefault("diagnostic_stride", 5)
    nb["weight_safety"] = safety_cfg

    gate_cfg = deepcopy(nb.get("supervisor_gate", {}))
    gate_cfg["candidate_mode"] = "deterministic"
    gate_cfg["advantage_margin"] = 0.5
    gate_cfg["critic_dominance_gate_enabled"] = True
    gate_cfg["critic_dominance_margin"] = 0.5
    gate_cfg["score_uncertainty_weight"] = 0.5
    gate_cfg["score_supervisor_action_weight"] = 0.05
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
        str(BASE_WEIGHTS_RUNNER),
        run_name="__main__",
        init_globals={
            "NB_CONFIGURE": configure_sg_sac_weights_critic_warm,
            "NOTEBOOK_SOURCE_OVERRIDE": THIS_RUNNER,
            "RUN_SUMMARY_TITLE_OVERRIDE": "Distillation Weight SG-SAC Detgate Hidden-7 run summary",
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
