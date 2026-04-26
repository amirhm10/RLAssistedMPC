# Distillation Matrix Step 2 Wide Defaults

## Summary

The distillation plain matrix notebook was already wired to run the Step 1 scalar sensitivity diagnostic and pass Step 2 release-protected advisory caps into `utils.matrix_runner`. This update restores the plain distillation matrix action bounds to the wide raw search range while keeping Step 2 enabled by default.

## Changes

- Restored plain distillation matrix `alpha` raw action bounds to `0.75` through the capped wide upper value `min(1.25, alpha_cap)`.
- Kept plain matrix `B` multiplier bounds at the wide `0.75` to `1.25` range.
- Left Step 3 MPC acceptance/fallback disabled by default.
- Left structured-matrix A overrides unchanged; this change targets the plain matrix notebook/default path.

## Validation

- Validate notebook JSON for `distillation_RL_assisted_MPC_matrices_unified.ipynb`.
- Import distillation defaults and assert Step 2 is enabled, Step 3 is disabled, and plain matrix bounds are wide.
