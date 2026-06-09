# 2026-06-09 Distillation Active Runner Cleanup

## Summary

The distillation runner surface was consolidated to match the simpler polymer workflow. Active root entrypoints now cover only offset-free MPC, horizon, Markov, weights, and residual runs.

## Changes

- Moved inactive SAC, SG-SAC, TD7, dueling DQN, and wrapper-specific SG distillation runners to `archive/distillation_inactive_algorithm_entrypoints_20260609/`.
- Added `resolve_distillation_agent_kind(family, agent_mode)` with active modes `sg` and `plain`.
- Made SG the default active distillation mode: horizon resolves to `sg_dqn`, and Markov/weights/residual resolve to `sg_td3`.
- Kept plain mode available: horizon resolves to `dqn`, and Markov/weights/residual resolve to `td3`.
- Simplified future distillation RL result prefixes to `distillation_<family>_<sg|plain>_<run_mode>_<disturbance_profile>`.

## Preservation

Existing raw outputs under `Data/`, `Result/`, and `Results/` were not renamed or regenerated.
