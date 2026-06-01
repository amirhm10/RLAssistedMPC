"""Distillation SG-TD3 residual runner with critic-warm release.

This entrypoint is a clean ablation of the standard distillation residual
runner. It keeps the zero-residual supervisor gate, but disables behavioral
cloning, handoff blending, TD3 authority caps, rho authority, shadow residual
diagnostics, and early-release guards.

Training process:
- During the 10 warm-start episodes, the executed residual is zero, so the plant
  follows OF-MPC.
- During the next 3 episodes, the runner still executes the supervisor action
  while replay is collected and critic updates occur.
- After that critic-only window, the actor may train and the gate chooses the
  policy action only when its conservative critic score beats the zero-residual
  supervisor by a positive margin.
"""

from __future__ import annotations

import runpy
from copy import deepcopy
from pathlib import Path


THIS_RUNNER = Path(__file__).name
BASE_RESIDUAL_RUNNER = Path(__file__).with_name("distillation_RL_assisted_MPC_residual_unified.py")


def configure_sg_td3_residual_critic_warm(nb: dict) -> dict:
    """Apply runner-local SG-TD3 critic-warm settings for distillation."""
    nb = deepcopy(nb)

    nb["agent_kind"] = "sg_td3"
    nb["run_mode"] = "disturb"
    nb["disturbance_profile"] = "fluctuation"
    nb["state_mode"] = "mismatch"
    nb["warm_start_override"] = 10
    nb["post_warm_start_action_freeze_subepisodes"] = 3
    nb["post_warm_start_actor_freeze_subepisodes"] = 3
    nb["result_prefix_override"] = (
        "distillation_residual_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch_no_rho"
    )
    nb["compare_prefix_override"] = (
        "distillation_compare_residual_sg_td3_critic_warm3_manual_off_disturb_fluctuation"
    )

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

    safety_cfg = deepcopy(nb.get("residual_safety", {}))
    safety_cfg["enabled"] = True
    safety_cfg.setdefault("reward_probation", {})
    safety_cfg["reward_probation"]["enabled"] = False
    safety_cfg["fallback_to_zero_on_nonfinite"] = True
    for key in ("shadow_rho_authority", "shadow_residual_deadband", "shadow_direction_risk"):
        safety_cfg.setdefault(key, {})
        safety_cfg[key]["enabled"] = False
    safety_cfg.setdefault("early_release_guard", {})
    safety_cfg["early_release_guard"]["enabled"] = False
    nb["residual_safety"] = safety_cfg

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
        str(BASE_RESIDUAL_RUNNER),
        run_name="__main__",
        init_globals={
            "NB_CONFIGURE": configure_sg_td3_residual_critic_warm,
            "NOTEBOOK_SOURCE_OVERRIDE": THIS_RUNNER,
            "RUN_SUMMARY_TITLE_OVERRIDE": "Distillation Residual SG-TD3 Critic-Warm-3 Manual-Off run summary",
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
