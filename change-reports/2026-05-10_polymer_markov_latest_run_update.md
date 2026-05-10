# 2026-05-10 Polymer Markov Latest-Run Update

## What changed

- added `report/scripts/generate_polymer_markov_latest_run_update_20260510.py`
- generated `report/figures/polymer_markov_latest_run_20260510/`
- extended `report/polymer_markov_latest_run_analysis_2026_05_09.md` with the newest polymer Markov run from `Polymer/Results/td3_markov_disturb/20260510_193643/`

## Why

The May 9 report no longer reflected the newest shared-reward polymer Markov result. The latest run forces TD3 execution almost everywhere, so it is a cleaner test of whether the learned Markov correction is helping or only being masked by fallback logic.

## Main finding

The newest shared-reward polymer Markov run remains worse than canonical MPC on the shared reward despite `td3_fraction = 0.9999`, `ls_fraction = 0.0`, and `nominal_fraction = 0.0`. This strengthens the conclusion that the current polymer `io_pair_gain` Markov branch is not yet helping the process objective in a robust way.

## Verification

- ran `report/scripts/generate_polymer_markov_latest_run_update_20260510.py`
- confirmed creation of:
  - `report/figures/polymer_markov_latest_run_20260510/reward_mode_history.png`
  - `report/figures/polymer_markov_latest_run_20260510/shared_reward_window_compare.png`
  - `report/figures/polymer_markov_latest_run_20260510/recent_run_summary.csv`
  - `report/figures/polymer_markov_latest_run_20260510/summary.json`
