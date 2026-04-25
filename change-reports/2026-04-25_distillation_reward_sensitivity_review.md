# Distillation Reward Sensitivity Review

Date: 2026-04-25

## Summary

Corrected the report provenance for the latest distillation matrix run and added a reward-sensitivity analysis for the latest distillation residual and matrix trajectories.

## Corrections

- The latest distillation matrix TD3 run was not a Step 2 or Step 3 run.
- The run has no release-guard or MPC-acceptance logs.
- It is the pre-protection low-noise run reported by the user, with tight `A`, wide `B`, and exploration/policy smoothing noise set to `0.01`.

## Analysis

- Re-scored the latest residual and matrix trajectories under 10 reward candidates, including pure quadratic rewards.
- Residual can score better than MPC under pure quadratic and no-bonus rewards, even though it loses under the current bonus-shaped reward.
- Matrix remains poor over the full run under all tested rewards, but its tail improves x24 composition under output-only quadratic scoring.
- The matrix tail improvement is not enough to accept the run because temperature tracking and input movement remain worse.

## Files

- `report/matrix_multiplier_cap_calculation_and_distillation_recovery.md`
- `report/figures/distillation_reward_sensitivity_20260425/distillation_reward_candidate_full_tail_delta.png`
- `report/figures/distillation_reward_sensitivity_20260425/residual_reward_window_heatmap.png`
- `report/figures/distillation_reward_sensitivity_20260425/matrix_pre_step_reward_window_heatmap.png`
- `report/figures/distillation_reward_sensitivity_20260425/distillation_physical_good_enough_metrics.png`
- `report/figures/distillation_reward_sensitivity_20260425/reward_sensitivity_summary.csv`
- `report/figures/distillation_reward_sensitivity_20260425/physical_metric_summary.csv`
