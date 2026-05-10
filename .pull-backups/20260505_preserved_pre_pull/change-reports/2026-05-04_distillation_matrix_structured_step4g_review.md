# Distillation Matrix Step 4G Review

Date: 2026-05-04

## Summary

Created a research report for the latest full distillation TD3 scalar-matrix and structured-matrix runs with Step 2 release guarding and Step 4G behavioral cloning active. The review compares both RL-assisted runs against the disturbance-fluctuation MPC baseline and includes new reward, tracking, final-episode, and multiplier diagnostics.

## Main Finding

The saved May 3 full-rollout runs completed without nonfinite actions or structured-model fallback events, but neither controller improved over disturbance MPC. The scalar matrix run degraded strongly late in training, while the structured matrix run recovered better in the final episode but still underperformed the MPC reference.

## Files Added

- `report/distillation_matrix_structured_step4g_latest_2026_05_04.md`
- `report/figures/distillation_matrix_structured_step4g_20260504/`

## Verification

- Loaded the scalar matrix, structured matrix, comparison, and MPC baseline result bundles.
- Recomputed summary metrics from the saved reward/output/input/action traces.
- Copied original run plots into the report figure folder for auditability.
- Confirmed the saved runs use `decision_interval = 1`, so the recommended next experiment is the newly added distillation default `decision_interval = 20`.
