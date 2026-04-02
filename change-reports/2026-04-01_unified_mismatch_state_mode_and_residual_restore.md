# Unified Mismatch-State Mode And Residual Mismatch Restoration

## Scope

- Added `STATE_MODE = "standard" | "mismatch"` to all four unified notebook entrypoints:
  - `RL_assisted_MPC_horizons_unified.ipynb`
  - `RL_assisted_MPC_matrices_unified.ipynb`
  - `RL_assisted_MPC_weights_unified.ipynb`
  - `RL_assisted_MPC_residual_unified.ipynb`
- Added the shared RL state builder in `utils/state_features.py`.
- Updated all shared runners to consume the new state builder and propagate mismatch diagnostics.
- Restored the base legacy residual mismatch authority logic in `utils/residual_runner.py` when `state_mode == "mismatch"`.

## Main Changes

### Shared RL state mode

- `utils/state_features.py`
  - `get_rl_state_dim(...)`
  - `build_rl_state(...)`
- `standard` mode keeps the existing unified state:
  - augmented observer state
  - current setpoint
  - current input deviation
- `mismatch` mode appends the legacy mismatch features:
  - `innov = y_prev_scaled - yhat_pred`
  - `e_track = y_prev_scaled - y_sp`

### Shared runner updates

- Updated:
  - `utils/horizon_runner.py`
  - `utils/matrix_runner.py`
  - `utils/weights_runner.py`
  - `utils/residual_runner.py`
- Each runner now:
  - accepts `state_mode` from the notebook config
  - builds current and next RL states through `utils.state_features`
  - stores `state_mode` in the normalized result bundle
  - stores `innovation_log` and `tracking_error_log` when mismatch mode is active

### Residual mismatch restoration

- `utils/residual_runner.py` now has two explicit paths:
  - `state_mode == "standard"`
    - keeps the clip-only residual behavior introduced in the unified residual runtime
  - `state_mode == "mismatch"`
    - restores the base legacy residual mismatch idea from `RL_assisted_MPC_residual_model_mismatch.ipynb`
- Restored mismatch-mode residual authority terms:
  - `band_scaled`
  - `eps_i = eta_tol * band_scaled`
  - `rho = max(|e_track| / eps_i)`
  - `mag = (rho * beta_res) * (abs(delta_u_mpc) + du0_res)`
- The replay action remains the executed residual action, not the raw proposal.
- Plant-side mismatch schedules such as `CMf` were not reintroduced into the unified notebooks.

### Plotting

- `utils/plotting_core.py` now normalizes:
  - `state_mode`
  - `innovation_log`
  - `tracking_error_log`
- Added mismatch diagnostic plots for unified runs when those logs are present.

### Notebook updates

- All unified notebooks now:
  - expose `STATE_MODE` in the first config cell
  - compute `STATE_DIM` through `get_rl_state_dim(...)`
  - pass `state_mode` into the shared runner config
  - keep result/comparison output directories distinct by appending the state mode suffix

## Validation

Validated in `rl-env` with shortened live smoke runs:

- Notebook schema validation:
  - all four unified notebooks
- Module compile/import validation:
  - `utils/state_features.py`
  - updated runners
  - updated plotting layer
- Mismatch-mode live smoke runs:
  - horizons: nominal, disturb
  - matrices: TD3 nominal/disturb, SAC nominal/disturb
  - weights: TD3 nominal/disturb, SAC nominal/disturb
  - residual: TD3 nominal/disturb, SAC nominal/disturb
- Residual-specific checks:
  - mismatch state dimension matches the legacy formula
  - `innovation_log` and `tracking_error_log` are populated
  - replay stores the executed residual action
  - mismatch-mode executed residuals satisfy the restored `rho`-scaled magnitude bound
  - `STATE_MODE="standard"` still runs without mismatch logs

## Files Changed

- `utils/state_features.py`
- `utils/horizon_runner.py`
- `utils/matrix_runner.py`
- `utils/weights_runner.py`
- `utils/residual_runner.py`
- `utils/plotting_core.py`
- `RL_assisted_MPC_horizons_unified.ipynb`
- `RL_assisted_MPC_matrices_unified.ipynb`
- `RL_assisted_MPC_weights_unified.ipynb`
- `RL_assisted_MPC_residual_unified.ipynb`
- `report/notebook_refactor_audit.md`
