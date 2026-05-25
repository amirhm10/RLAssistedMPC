# Residual Rho-Off Logging Fix

Date: 2026-05-25

## Summary

Fixed a crash in the standalone residual runner when residual rho authority is disabled. With `residual_authority_enabled=False`, `project_residual_action` correctly returns `rho=None`, `rho_raw=None`, and `rho_eff=None`, but the runner was still casting those fields to floats for mismatch-mode logs.

## Change

- Initialize `rho_log`, `rho_raw_log`, and `rho_eff_log` as `NaN` arrays.
- Record `NaN` when rho projection fields are `None`.
- This keeps the rho-off experiment valid: rho is not used as a state feature and rho authority is not applied.

## Validation

- Compile check for `utils/residual_runner.py`.
- Config check confirmed standalone distillation residual has:
  - `append_rho_to_state=False`
  - `residual_authority_enabled=False`
  - `authority_use_rho=False`
  - `use_rho_authority=False`
