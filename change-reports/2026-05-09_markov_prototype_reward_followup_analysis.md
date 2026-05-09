# 2026-05-09 Markov Prototype-Reward Follow-up Analysis

## What changed

- Added `report/scripts/analyze_polymer_markov_prototype_reward_followup.py`.
- Generated a new follow-up figure folder:
  `report/figures/polymer_markov_prototype_reward_followup_20260509/`
- Extended `report/polymer_markov_latest_run_analysis_2026_05_09.md` with a new section covering the newest prototype-reward unified rerun.

## Main findings captured

- The newest unified rerun does beat the canonical baseline under the prototype reward.
- That gain is dominated by the old exponential inside-band bonus, not by a large change in visible tracking or input movement.
- The earlier unified run already looked good under the prototype reward when rescored, so the reward switch changed evaluation more than it changed the actual controller behavior.
- The remaining difference from the previous prototype run comes mainly from the unified nominal/comparison path and less aggressive correction usage, not from `z_bound`.
