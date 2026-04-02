# Residual Rho Toggle

## Scope

- Confirmed and preserved the current residual behavior:
  - `STATE_MODE="standard"` does not use `rho`
  - `STATE_MODE="mismatch"` previously always used `rho`
- Added an explicit residual mismatch toggle so `rho`-scaled authority can be enabled or disabled from the unified residual notebook.

## Changes

- Updated `utils/residual_runner.py`
  - new config flag: `use_rho_authority`
  - default remains `True` to preserve the existing mismatch behavior
  - in mismatch mode:
    - if `use_rho_authority=True`, residual authority uses the restored legacy form
      - `mag = (rho * beta_res) * (abs(delta_u_mpc) + du0_res)`
    - if `use_rho_authority=False`, residual authority keeps the same mismatch-state path but removes the `rho` scaling term
      - `mag = beta_res * (abs(delta_u_mpc) + du0_res)`
  - added result-bundle fields:
    - `use_rho_authority`
    - `rho_log`
- Updated `RL_assisted_MPC_residual_unified.ipynb`
  - new top-level switch:
    - `USE_RHO_AUTHORITY = True`
  - passes `use_rho_authority` into `residual_cfg`
  - appends `_rho` or `_no_rho` to mismatch-mode result and comparison directory prefixes so runs do not overwrite each other

## Validation

Validated in `rl-env` with:

- `utils/residual_runner.py` compile check
- `RL_assisted_MPC_residual_unified.ipynb` schema validation
- shortened residual smoke runs for:
  - `STATE_MODE="standard"`
  - `STATE_MODE="mismatch", USE_RHO_AUTHORITY=True`
  - `STATE_MODE="mismatch", USE_RHO_AUTHORITY=False`
- confirmed:
  - standard mode still has no `rho_log`
  - mismatch mode logs `rho_log`
  - the new toggle is exposed in the notebook and propagated through the runner
