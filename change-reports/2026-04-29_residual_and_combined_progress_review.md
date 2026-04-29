# Residual And Combined Progress Review

Date: 2026-04-29

## Summary

Added a focused research report analyzing saved polymer residual, polymer combined residual-active, and distillation residual runs.

## Main Findings

- Polymer residual improves reward under disturbance, but the saved disturbance runs do not beat the disturbance MPC baseline in physical tail MAE.
- Later nominal polymer residual runs remove the nominal tail offset.
- Polymer combined residual-active runs are the strongest reward results, but they also show a small offset and larger tail MAE than disturbance MPC.
- Distillation residual runs show the reported pattern: reward can surpass the MPC reward reference, then degrade later, and a non-negligible late offset can appear.
- Projection is active almost continuously in mismatch residual runs, so raw actor actions are much larger than executed residual actions.

## Files Changed

- `report/residual_and_combined_progress_2026_04_29.md`
- `change-reports/2026-04-29_residual_and_combined_progress_review.md`

## Verification

Metrics were computed from saved `input_data.pkl` bundles using physical setpoint reconstruction from the saved scaling artifacts.
