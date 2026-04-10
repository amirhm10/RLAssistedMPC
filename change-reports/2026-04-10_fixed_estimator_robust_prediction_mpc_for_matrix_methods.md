# Fixed Estimator Robust Prediction MPC For Matrix Methods

## Scope
- Updated scalar matrix, structured matrix, and combined matrix paths
- Covered both polymer and distillation unified notebooks
- Left horizon, weights, residual, rewards, and plant models unchanged

## Main Behavior Change
- Matrix-assisted model updates now affect MPC prediction only
- State estimation now stays on the fixed nominal observer path:
  - nominal `A_est`
  - nominal `B_est`
  - fixed observer gain `L0`
- The old `recalculate_observer_on_matrix_change` flag is retained only as a backward-compatible ignored setting for matrix methods

## Robust MPC Layer
- Added `utils/robust_matrix_prediction.py`
- Added:
  - input-bound tightening
  - candidate model validation
  - short nominal-vs-assisted prediction screening
  - assisted-to-nominal solver fallback
  - multiplier sensitivity sweep helper
- Fallback order:
  1. assisted model with tightened bounds
  2. nominal model with tightened bounds
  3. nominal model with original bounds

## Updated Shared Runners
- `utils/matrix_runner.py`
- `utils/structured_matrix_runner.py`
- `utils/combined_runner.py`

## Notebook / Default Updates
- Added visible robust-prediction controls to:
  - `RL_assisted_MPC_matrices_unified.ipynb`
  - `RL_assisted_MPC_structured_matrices_unified.ipynb`
  - `RL_assisted_MPC_combined_unified.ipynb`
  - `distillation_RL_assisted_MPC_matrices_unified.ipynb`
  - `distillation_RL_assisted_MPC_structured_matrices_unified.ipynb`
  - `distillation_RL_assisted_MPC_combined_unified.ipynb`
- Added shared defaults in:
  - `systems/polymer/notebook_params.py`
  - `systems/distillation/notebook_params.py`

## New Notebook-Facing Controls
- `INPUT_TIGHTENING_FRAC`
- `ENABLE_ACCEPT_NORM_TEST`
- `EPS_A_NORM_FRAC`
- `EPS_B_NORM_FRAC`
- `ENABLE_ACCEPT_PREDICTION_TEST`
- `PREDICTION_CHECK_HORIZON`
- `EPS_Y_PRED_SCALED`
- `ENABLE_SOLVER_FALLBACK`
- `PROBE_INPUT_MODE`

## Plotting / Saved Bundle Updates
- Extended matrix and combined plotting to show optional:
  - acceptance/source traces
  - prediction-screening metrics
  - rejection reason counts
- Saved bundles now carry the robust matrix diagnostics so plots can be regenerated from `input_data.pkl`

## Validation
- `py_compile` on touched Python modules
- AST parsing on the six touched matrix / structured-matrix / combined notebooks
- `rl-env` smoke check on the shared robust helper
