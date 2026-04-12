# Remove Matrix Acceptance/Fallback Layer And Always Use Assisted Prediction

## Summary

Removed the matrix acceptance, input-tightening, and nominal-fallback layer from the active matrix-based methods while keeping the fixed nominal estimator split in place.

Affected paths:
- polymer scalar matrix
- polymer structured matrix
- polymer combined matrix branch
- distillation scalar matrix
- distillation structured matrix
- distillation combined matrix branch

Post-change behavior:
- the estimator remains nominal and fixed
- the RL-updated matrix candidate is used directly for MPC prediction
- original MPC input bounds are used
- unsuccessful assisted MPC solves now raise immediately instead of silently falling back

## Code Changes

Updated runners:
- `utils/matrix_runner.py`
- `utils/structured_matrix_runner.py`
- `utils/combined_runner.py`

What changed in the runners:
- removed candidate validation
- removed tightened-bound construction
- removed nominal fallback sequencing
- removed acceptance and rejection bookkeeping
- preserved nominal estimator recursion with fixed `A_est`, `B_est`, and `L_nom`
- preserved scalar and structured matrix action semantics
- preserved combined `decision_interval` holding for the matrix branch

Deleted unused helper:
- `utils/robust_matrix_prediction.py`

Plotting cleanup:
- `utils/plotting_core.py`

Removed new-bundle dependence on:
- acceptance logs
- fallback/source logs
- rejection counts
- tightened-bound metadata
- prediction-screening metrics

## Notebook And Defaults Cleanup

Removed the robust-prediction notebook surface from:
- `RL_assisted_MPC_matrices_unified.ipynb`
- `RL_assisted_MPC_structured_matrices_unified.ipynb`
- `RL_assisted_MPC_combined_unified.ipynb`
- `distillation_RL_assisted_MPC_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_structured_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_combined_unified.ipynb`

Removed the corresponding defaults from:
- `systems/polymer/notebook_params.py`
- `systems/distillation/notebook_params.py`

Removed notebook-facing controls:
- `INPUT_TIGHTENING_FRAC`
- `ENABLE_ACCEPT_NORM_TEST`
- `EPS_A_NORM_FRAC`
- `EPS_B_NORM_FRAC`
- `ENABLE_ACCEPT_PREDICTION_TEST`
- `PREDICTION_CHECK_HORIZON`
- `EPS_Y_PRED_SCALED`
- `ENABLE_SOLVER_FALLBACK`
- `PROBE_INPUT_MODE`

## Validation

Validated with:
- repo search confirming the active matrix paths no longer reference the removed acceptance/fallback helpers or config keys
- syntax compilation of the touched runner, plotting, and notebook-default modules
- AST parsing of the six touched notebooks

## Notes

- The fixed-estimator architecture remains intact by design.
- Old result bundles can still be opened by existing plotting entrypoints; the removed acceptance/fallback fields are now simply ignored when absent.
