# 2026-05-13 Distillation Markov Variant Notebooks

## Scope

- added `distillation_RL_assisted_MPC_markov_ls_only_unified.ipynb`
- added `distillation_RL_assisted_MPC_markov_td3_without_ls_unified.ipynb`
- added `distillation_RL_assisted_MPC_markov_relaxed_acceptance_unified.ipynb`

## Purpose

These are runnable copies of the canonical distillation Markov notebook for three targeted follow-up experiments:

- LS-only
- TD3 without LS
- relaxed acceptance thresholds

## Aspen mapping

All three copies stay on the distillation `markov` family and now explicitly set:

- `distillation_RL_assisted_MPC_markov_ls_only_unified.ipynb` -> Aspen simulation number `12` -> `C2S_SS_simulation12.dynf`
- `distillation_RL_assisted_MPC_markov_td3_without_ls_unified.ipynb` -> Aspen simulation number `13` -> `C2S_SS_simulation13.dynf`
- `distillation_RL_assisted_MPC_markov_relaxed_acceptance_unified.ipynb` -> Aspen simulation number `14` -> `C2S_SS_simulation14.dynf`

## Notes

- each notebook uses a unique `result_prefix_override`
- each notebook uses a unique `compare_prefix_override`
- outputs were cleared so the copies can be run cleanly
