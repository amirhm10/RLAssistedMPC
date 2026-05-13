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

All three copies stay on the distillation `markov` family and explicitly set:

- Aspen simulation number `11`
- dyn file `C2S_SS_simulation11.dynf`

## Notes

- each notebook uses a unique `result_prefix_override`
- each notebook uses a unique `compare_prefix_override`
- outputs were cleared so the copies can be run cleanly
