# Method-Scoped Conditioning And Residual Refresh

Date: 2026-04-20

## Summary

Implemented the planned method-scoped conditioning and residual refresh batch behind legacy-preserving flags.

## Main Changes

- Added [utils/observation_conditioning.py](../utils/observation_conditioning.py)
  - running z-score conditioning for physical `xhat`
  - mismatch-feature transforms: `hard_clip`, `soft_tanh`, `signed_log`
  - shared observer-alignment helper
- Extended [utils/state_features.py](../utils/state_features.py)
  - new mismatch/conditioning config resolution
  - raw mismatch feature logging
  - opt-in state conditioning hook
- Extended [utils/residual_authority.py](../utils/residual_authority.py)
  - raw-tracking-based `rho` computation
  - `exp_raw_tracking` mapping
  - near-setpoint residual deadband
- Updated mismatch-capable runners:
  - [utils/horizon_runner.py](../utils/horizon_runner.py)
  - [utils/horizon_runner_dueling.py](../utils/horizon_runner_dueling.py)
  - [utils/matrix_runner.py](../utils/matrix_runner.py)
  - [utils/structured_matrix_runner.py](../utils/structured_matrix_runner.py)
  - [utils/weights_runner.py](../utils/weights_runner.py)
  - [utils/residual_runner.py](../utils/residual_runner.py)
  - [utils/combined_runner.py](../utils/combined_runner.py)
  - [utils/reidentification_runner.py](../utils/reidentification_runner.py)
- Added notebook-default flags in:
  - [systems/polymer/notebook_params.py](../systems/polymer/notebook_params.py)
  - [systems/distillation/notebook_params.py](../systems/distillation/notebook_params.py)
- Refreshed the diagnostics/report layer:
  - [report/generate_rl_state_scaling_report.py](../report/generate_rl_state_scaling_report.py)
  - [report/rl_state_scaling_diagnostics.md](../report/rl_state_scaling_diagnostics.md)

## Validation

- `C:\Users\HAMEDI\miniconda3\envs\rl-env\python.exe -m compileall utils systems report\generate_rl_state_scaling_report.py`
- `C:\Users\HAMEDI\miniconda3\envs\rl-env\python.exe report\generate_rl_state_scaling_report.py`

## Notes

- Legacy behavior remains the default.
- The report now includes offline replay diagnostics for the new residual `rho` mapping plus deadband on the saved residual TD3 trajectories.
