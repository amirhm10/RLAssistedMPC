"""Polymer SG-TD3 Markov runner with base-only standard RL observations.

This ablation keeps the LS-or-MPC Markov supervisor and action space, but feeds
the policy only the standard base RL state. It omits Markov-specific state
features such as previous correction, LS correction, prediction score, and gain
drift from the agent input.
"""

from __future__ import annotations

import runpy
from copy import deepcopy
from pathlib import Path

from RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_unified import (
    configure_sg_td3_markov_critic_warm,
)


THIS_RUNNER = Path(__file__).name
BASE_MARKOV_RUNNER = Path(__file__).with_name("RL_assisted_MPC_markov_unified.py")


def configure_sg_td3_markov_critic_warm_standard_standard(nb: dict) -> dict:
    """Apply SG-TD3 Markov settings with base-only standard policy input."""
    nb = configure_sg_td3_markov_critic_warm(nb)
    nb = deepcopy(nb)

    nb["state_mode"] = "standard"
    nb["markov_agent_state_features"] = "base_only"
    nb["result_prefix_override"] = "sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_standard_standard"
    nb["compare_prefix_override"] = (
        "disturb_compare_sg_td3_markov_critic_warm3_ls_else_mpc_shadow_standard_standard"
    )

    profiles = deepcopy(nb.get("run_profiles", {}))
    profiles[("sg_td3", "nominal")] = {
        "result_prefix": "sg_td3_markov_critic_warm3_ls_else_mpc_shadow_nominal_standard_standard",
        "compare_prefix": "nominal_compare_sg_td3_markov_critic_warm3_ls_else_mpc_shadow_standard_standard",
        "compare_mode": "nominal",
        "plot_start_episode": 2,
        "compare_start_episode": 2,
    }
    profiles[("sg_td3", "disturb")] = {
        "result_prefix": "sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_standard_standard",
        "compare_prefix": "disturb_compare_sg_td3_markov_critic_warm3_ls_else_mpc_shadow_standard_standard",
        "compare_mode": "disturb",
        "plot_start_episode": 2,
        "compare_start_episode": 2,
    }
    nb["run_profiles"] = profiles
    return nb


def main() -> dict:
    run_globals = runpy.run_path(
        str(BASE_MARKOV_RUNNER),
        run_name="__main__",
        init_globals={
            "NB_CONFIGURE": configure_sg_td3_markov_critic_warm_standard_standard,
            "NOTEBOOK_SOURCE_OVERRIDE": THIS_RUNNER,
            "RUN_SUMMARY_TITLE_OVERRIDE": "Polymer Markov SG-TD3 Base-Only Standard run summary",
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
