# Polymer Structured Step 3D Default

Date: 2026-04-29

## Summary

Changed the shared polymer `structured_matrix` notebook default from Step 3C shadow mode to Step 3D usefulness-gated execution so it matches the distillation structured path.

## Changes

- Left the polymer scalar `matrix` family on Step 3C shadow mode.
- Kept behavioral cloning disabled for polymer `structured_matrix`.
- Disabled `mpc_dual_cost_shadow` for polymer `structured_matrix`.
- Enabled `mpc_usefulness_gate` for polymer `structured_matrix`.
- Left Step 2 release-protected advisory caps enabled.
- Left acceptance fallback disabled.

## Validation

- Imported `get_polymer_notebook_defaults("matrix")` and `get_polymer_notebook_defaults("structured_matrix")` in `rl-env`.
- Confirmed `matrix` resolves to `Step 3C study (Step 2 on, Step 4 off)`.
- Confirmed `structured_matrix` resolves to `Step 3D (Step 2 + usefulness gate)`.
- Ran `py_compile` on `systems/polymer/notebook_params.py`.
