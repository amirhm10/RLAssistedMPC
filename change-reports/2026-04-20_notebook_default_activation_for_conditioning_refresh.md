# Notebook Default Activation For Conditioning Refresh

Date: 2026-04-20

## Summary

Activated the previously implemented conditioning/residual-refresh features as the default behavior of the unified RL notebooks for both polymer and distillation.

## What Changed

- updated polymer and distillation notebook default tables so RL notebooks open in disturbance + mismatch mode by default
- set polymer mismatch conditioning defaults to running `xhat` normalization, signed-log mismatch transforms, and current-measurement observer alignment
- set distillation mismatch conditioning defaults to signed-log mismatch transforms and current-measurement observer alignment, while keeping fixed min-max base-state scaling
- set residual-family defaults to `rho_mapping_mode="exp_raw_tracking"` with deadband enabled
- patched all unified RL notebooks so they actually pass the new conditioning and residual-authority fields into the shared runner configs
- removed the hard-coded `RUN_MODE = "disturb"` override from the polymer residual notebook so it now follows the shared notebook defaults cleanly

## Files

- [systems/polymer/notebook_params.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/polymer/notebook_params.py)
- [systems/distillation/notebook_params.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/distillation/notebook_params.py)
- unified RL notebooks under the repo root
- [report/method_scoped_conditioning_change_log.md](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/report/method_scoped_conditioning_change_log.md)

## Validation

- imported both notebook-default modules and checked the resolved defaults
- re-read every edited notebook with `nbformat`
- verified the new config keys are present in the notebook code cells
