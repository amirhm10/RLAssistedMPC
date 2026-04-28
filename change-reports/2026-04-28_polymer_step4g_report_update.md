# 2026-04-28 Polymer Step 4G Report Update

## Summary

Extended the matrix multiplier report with analysis of the latest polymer Step 4G reruns for scalar and structured matrix supervisors.

## What Changed

- Added a new Step 4G results section to `report/matrix_multiplier_cap_calculation_and_distillation_recovery.md`.
- Compared:
  - scalar Step 4G against the stronger scalar BC-only baseline
  - structured Step 4G against the weighted Step 4E baseline
- Added new figure and CSV assets under `report/figures/matrix_multiplier_step4g_20260428/`.
- Added a reproducible asset-generation script:
  - `report/scripts/generate_polymer_step4g_update.py`

## Main Findings Captured

- Scalar Step 4G materially improves the first live windows and slightly improves the full-run reward over the stronger scalar BC-only baseline.
- Structured Step 4G greatly improves the early release windows over weighted Step 4E, but gives back some later reward because the current guard timing is still conservative.
- Both runs remain clean Step 4G readouts:
  - release guard active
  - no acceptance fallback
