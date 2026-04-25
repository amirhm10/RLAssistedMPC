# Analyze Step 3B And Latest Distillation Results

Date: 2026-04-25

## Summary

Updated the ongoing matrix-multiplier report with the latest polymer Step 3B result and the latest distillation residual and matrix runs.

## Findings

- Polymer Step 3B increased live acceptance compared with strict Step 3, but did not improve performance.
- Scalar matrix Step 3B accepted roughly half of live decisions and still lost to MPC over the full run.
- Structured matrix Step 3B improved over strict Step 3 in the middle window, but lost most of the Step 2-only tail benefit.
- The latest distillation matrix run, before the new steps were active, had a severe release crash despite tight `A` bounds.
- The latest distillation residual run was much safer than matrix, but still lost overall and degraded in the tail.

## Recommendation

Do not keep increasing Step 3B tolerance. The next step should be Step 3C as a shadow-only dual-cost diagnostic: log both nominal safety penalty and candidate-model advantage before using the gate for fallback.

## Files

- `report/matrix_multiplier_cap_calculation_and_distillation_recovery.md`
- `report/figures/matrix_multiplier_latest_cross_system_20260425/polymer_step3b_reward_delta_comparison.png`
- `report/figures/matrix_multiplier_latest_cross_system_20260425/polymer_step3b_gate_acceptance.png`
- `report/figures/matrix_multiplier_latest_cross_system_20260425/distillation_latest_reward_delta.png`
- `report/figures/matrix_multiplier_latest_cross_system_20260425/distillation_latest_diagnostics.png`
