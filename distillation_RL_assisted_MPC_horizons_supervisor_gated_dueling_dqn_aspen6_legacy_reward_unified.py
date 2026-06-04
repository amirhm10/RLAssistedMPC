"""Distillation SG-dueling-DQN horizon check with Aspen 6 and legacy reward.

This is a duplicate check-runner for the SG-dueling-DQN horizon workflow. It
keeps the OF-MPC supervisor gate and critic-warm settings from the main
distillation SG-dueling runner, but forces `C2S_SS_simulation6.dynf` via
`aspen_preset = 6` and restores the archived horizon reward parameters.
"""

from __future__ import annotations

import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np

from DuelingDQN.supervisor_gated_dueling_dqn_agent import SupervisorGatedDuelingDQNAgent


THIS_RUNNER = Path(__file__).name
BASE_DUELING_RUNNER = Path(__file__).with_name("distillation_RL_assisted_MPC_horizons_dueling_unified.py")


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


def _restore_legacy_horizon_reward(nb: dict) -> None:
    reward_cfg = deepcopy(nb.get("reward", {}))
    reward_cfg.update(
        {
            "k_rel": np.asarray([0.3, 0.02], dtype=float),
            "band_floor_phys": np.asarray([0.003, 0.3], dtype=float),
            "Q_diag": np.asarray([3.7e4, 1.5e3], dtype=float),
            "R_diag": np.asarray([2.5e3, 2.5e3], dtype=float),
            "tau_frac": 0.7,
            "gamma_out": 0.5,
            "gamma_in": 0.5,
            "beta": 7.0,
            "gate": "geom",
            "lam_in": 1.0,
            "bonus_kind": "exp",
            "bonus_k": 12.0,
            "bonus_p": 0.6,
            "bonus_c": 20.0,
            "reward_scale": 1.0,
        }
    )
    nb["reward"] = reward_cfg


def configure_sg_dueling_dqn_horizon_aspen6_legacy_reward(nb: dict) -> dict:
    """Apply SG-dueling-DQN settings with Aspen 6 and archived horizon reward."""
    nb = deepcopy(nb)

    nb["agent_kind"] = "sg_dueling_dqn"
    nb["run_mode"] = "disturb"
    nb["disturbance_profile"] = "fluctuation"
    nb["state_mode"] = "standard"
    nb["aspen_preset"] = 6
    nb["warm_start_override"] = 10
    nb["post_warm_start_action_freeze_subepisodes"] = 3
    nb["result_prefix_override"] = (
        "distillation_dueling_horizon_sg_dqn_aspen6_legacyreward_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_standard_np6_11_nc3_11"
    )
    nb["compare_prefix_override"] = (
        "distillation_compare_dueling_horizon_sg_dqn_aspen6_legacyreward_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_standard_np6_11_nc3_11"
    )

    ctrl_cfg = deepcopy(nb.get("controller", {}))
    ctrl_cfg["predict_grid"] = list(range(6, 12))
    ctrl_cfg["control_grid"] = list(range(3, 12))
    nb["controller"] = ctrl_cfg

    _disable_horizon_safety(nb)
    _restore_legacy_horizon_reward(nb)

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
    agent_cfg["multistep_mode"] = "n_step"
    agent_cfg["supervisor_gate"] = deepcopy(gate_cfg)
    nb["agent"] = agent_cfg

    return nb


def main() -> dict:
    run_globals = runpy.run_path(
        str(BASE_DUELING_RUNNER),
        run_name="__main__",
        init_globals={
            "NB_CONFIGURE": configure_sg_dueling_dqn_horizon_aspen6_legacy_reward,
            "NOTEBOOK_SOURCE_OVERRIDE": THIS_RUNNER,
            "RUN_SUMMARY_TITLE_OVERRIDE": "Distillation Dueling Horizon SG-DQN Aspen-6 Legacy-Reward run summary",
            "HORIZON_AGENT_CLASS_OVERRIDE": SupervisorGatedDuelingDQNAgent,
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
