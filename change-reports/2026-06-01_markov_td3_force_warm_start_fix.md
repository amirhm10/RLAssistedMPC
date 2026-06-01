# Markov TD3 Force-Warm-Start Option

## Summary

Added an explicit Markov runner option so the distillation TD3-only no-safeguard current-reward variant can keep the configured warm-start action gate even when `force_td3_execute=True`.

In this specific current-reward variant, `run_adaptive_ls=False`, so the warm-start action gate means nominal zero-Markov-correction execution, not LS execution.

This is an intentional new current-reward benchmark behavior, not an exact restoration of the May 18 TD3-only run.

## Historical Pin-Down

- `change-reports/2026-05-16_add_distillation_td3_only_no_safeguard_notebook.md` added the no-safeguard notebook with `force_td3_execute=True`.
- `change-reports/2026-05-18_td3_priority_soft_release_markov.md` explicitly kept `force_td3_execute=True` behavior unchanged.
- `Distillation/Results/distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_unified/20260518_091937/input_data.pkl` has `force_td3_execute=True`, `warm_start_step=4000`, and `rl_action_source_log=2` (`td3_accepted`) from episode 1 through episode 10.
- The old reward dip begins around episode 11, so the old "warm-start" boundary was mainly a training/replay/BC timing boundary, not an executed-action LS warm start.

## Changes

- Added `force_td3_respects_warm_start` support to the shared Markov runner.
- Enabled that flag only in `distillation_RL_assisted_MPC_markov_td3_only_no_safeguard_current_reward_unified.py`.
- Logged the flag in the run summary and result bundle configuration.

## Expected Effect

For the current-reward no-safeguard variant, the first `warm_start` subepisodes should execute the nominal zero-Markov-correction action instead of reporting TD3 execution from subepisode 1. TD3-only forced execution still begins after the warm-start boundary.

For exact May 18 behavior reproduction, leave `force_td3_respects_warm_start=False`.

## Validation

- Static compile of the modified Markov runner and distillation TD3-only no-safeguard entrypoint.
- Static checks that the variant sets and passes `force_td3_respects_warm_start=True`.

## Runtime Notes

- No Aspen/distillation runtime was launched during this patch.
