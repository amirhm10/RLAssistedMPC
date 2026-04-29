# Polymer Step 3C Defaults

Date: 2026-04-29

## Summary

Switched the shared polymer `matrix` and `structured_matrix` notebook defaults from Step 4G to Step 3C shadow mode.

## Changes

- Disabled behavioral cloning in the polymer `matrix` defaults.
- Enabled `mpc_dual_cost_shadow` in the polymer `matrix` defaults.
- Disabled behavioral cloning in the polymer `structured_matrix` defaults.
- Enabled `mpc_dual_cost_shadow` in the polymer `structured_matrix` defaults.
- Left Step 2 release-protected advisory caps enabled in both families.
- Left Step 3D usefulness gating disabled in both families.
- Left acceptance fallback disabled in both families.

## Validation

- Imported `get_polymer_notebook_defaults("matrix")` and `get_polymer_notebook_defaults("structured_matrix")` in `rl-env`.
- Confirmed both families now resolve to `Step 3C study (Step 2 on, Step 4 off)`.
- Ran `py_compile` on `systems/polymer/notebook_params.py`.
