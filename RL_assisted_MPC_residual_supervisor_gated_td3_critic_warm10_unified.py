"""Polymer SG-TD3 residual runner with a 10-episode critic-only release.

This is the direct follow-up to the 5-episode critic-warm ablation. It keeps all
of the same clean SG-TD3 settings, but extends the post-warm zero-residual
critic-only phase from 5 to 10 subepisodes before actor release.
"""

from __future__ import annotations

import runpy
from pathlib import Path

from RL_assisted_MPC_residual_supervisor_gated_td3_critic_warm_unified import (
    configure_critic_warm_start,
)


THIS_RUNNER = Path(__file__).name
BASE_SG_TD3_RUNNER = Path(__file__).with_name("RL_assisted_MPC_residual_supervisor_gated_td3_unified.py")


def configure_critic_warm10_start(nb: dict) -> dict:
    """Apply the critic-warm settings with a 10-subepisode critic-only window."""
    nb = configure_critic_warm_start(nb)
    nb["post_warm_start_action_freeze_subepisodes"] = 10
    nb["post_warm_start_actor_freeze_subepisodes"] = 10
    nb["result_prefix_override"] = "sg_td3_residual_critic_warm10_disturb"
    nb["compare_prefix_override"] = "disturb_compare_sg_td3_residual_critic_warm10"
    return nb


def main() -> dict:
    run_globals = runpy.run_path(
        str(BASE_SG_TD3_RUNNER),
        run_name="__main__",
        init_globals={
            "NB_CONFIGURE": configure_critic_warm10_start,
            "NOTEBOOK_SOURCE_OVERRIDE": THIS_RUNNER,
            "RUN_SUMMARY_TITLE_OVERRIDE": "Polymer Residual SG-TD3 OF-MPC Warm-Start Critic-Only-10 run summary",
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
