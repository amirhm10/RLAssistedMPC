# Distillation Matrix Deep Review Report

Date: 2026-05-01

## Summary

Added a distillation-only matrix-family review that separates matched TD3 evidence from mismatched SAC compare bundles, identifies the late-improving `20260425_082831` TD3 B-only run as partial recovery rather than a success, and ends with a concrete stop-or-pivot experiment sequence.

## Files Changed

- `report/distillation_matrix_deep_review_2026_05_01.md`
- `report/scripts/generate_distillation_matrix_deep_review_assets.py`

## Generated Assets

- `report/figures/distillation_matrix_deep_review_20260501/distillation_matrix_deep_review_summary.csv`
- `report/figures/distillation_matrix_deep_review_20260501/distillation_matrix_window_summary.csv`
- `report/figures/distillation_matrix_deep_review_20260501/distillation_matrix_schedule_alignment.png`
- `report/figures/distillation_matrix_deep_review_20260501/distillation_matrix_family_reward_windows.png`
- `report/figures/distillation_matrix_deep_review_20260501/distillation_td3_tail_physical_tradeoff.png`

## Notes

- No controller or notebook execution logic was changed in this pass.
- The review documents a scientific validity risk in the saved SAC disturbance compare bundles because their RL schedules do not match the referenced disturbance MPC baseline schedule.
