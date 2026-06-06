"""Distillation SG-DQN horizon runner with OF-MPC supervisor gating.

This entrypoint is a thin wrapper around the standard distillation horizon
runner. It uses the default OF-MPC horizon `(6, 3)` as the supervisor action
for every SG-DQN gate comparison and leaves the older horizon safety layer
disabled so the first ablation isolates the learned Q gate.
"""

from __future__ import annotations

import runpy
from copy import deepcopy
from pathlib import Path

from DQN.supervisor_gated_dqn_agent import SupervisorGatedDQNAgent


THIS_RUNNER = Path(__file__).name
BASE_HORIZON_RUNNER = Path(__file__).with_name("distillation_RL_assisted_MPC_horizons_unified.py")


def _disable_horizon_safety(nb: dict) -> None:
    safety_cfg = deepcopy(nb.get("horizon_safety", {}))
    safety_cfg["enabled"] = False
    safety_cfg.setdefault("release_filter", {})
    safety_cfg["release_filter"]["enabled"] = False
    safety_cfg.setdefault("reward_probation", {})
    safety_cfg["reward_probation"]["enabled"] = False
    safety_cfg.setdefault("shadow_default_mpc", {})
    safety_cfg["shadow_default_mpc"]["enabled"] = False
    nb["horizon_safety"] = safety_cfg


def configure_sg_dqn_horizon_critic_warm(nb: dict) -> dict:
    """Apply runner-local SG-DQN settings for distillation horizon selection."""
    nb = deepcopy(nb)

    nb["agent_kind"] = "sg_dqn"
    nb["run_mode"] = "disturb"
    nb["disturbance_profile"] = "fluctuation"
    nb["state_mode"] = "mismatch"
    nb["warm_start_override"] = 10
    nb["post_warm_start_action_freeze_subepisodes"] = 5
    nb["result_prefix_override"] = (
        "distillation_horizon_sg_dqn_critic_warm5_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11"
    )
    nb["compare_prefix_override"] = (
        "distillation_compare_horizon_sg_dqn_critic_warm5_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11"
    )

    ctrl_cfg = deepcopy(nb.get("controller", {}))
    ctrl_cfg["predict_grid"] = list(range(6, 12))
    ctrl_cfg["control_grid"] = list(range(3, 12))
    nb["controller"] = ctrl_cfg

    _disable_horizon_safety(nb)

    gate_cfg = {
        "advantage_margin": 0.0,
        "default_to_supervisor": True,
        "min_train_steps_before_policy_gate": 0,
    }
    nb["supervisor_gate"] = gate_cfg

    agent_cfg = deepcopy(nb.get("agent", {}))
    agent_cfg["exploration_mode"] = "epsilon"
    agent_cfg["eps_start"] = 0.2
    agent_cfg["eps_end"] = 0.02
    agent_cfg["eps_decay_steps"] = 18_600
    agent_cfg["n_step"] = 1
    agent_cfg["multistep_mode"] = "one_step"
    agent_cfg["supervisor_gate"] = deepcopy(gate_cfg)
    nb["agent"] = agent_cfg

    return nb


def main() -> dict:
    run_globals = runpy.run_path(
        str(BASE_HORIZON_RUNNER),
        run_name="__main__",
        init_globals={
            "NB_CONFIGURE": configure_sg_dqn_horizon_critic_warm,
            "NOTEBOOK_SOURCE_OVERRIDE": THIS_RUNNER,
            "RUN_SUMMARY_TITLE_OVERRIDE": "Distillation Horizon SG-DQN Critic-Warm-5 OF-MPC run summary",
            "HORIZON_AGENT_CLASS_OVERRIDE": SupervisorGatedDQNAgent,
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
