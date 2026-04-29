# Polymer Matrix Step 3D Default

Date: 2026-04-29

## Summary

Changed the shared polymer scalar `matrix` notebook default from Step 3C shadow mode to Step 3D usefulness-gated execution so both polymer matrix families now use the same Step 2 + Step 3D path.

## Changes

- Kept behavioral cloning disabled for polymer `matrix`.
- Disabled `mpc_dual_cost_shadow` for polymer `matrix`.
- Enabled `mpc_usefulness_gate` for polymer `matrix`.
- Left Step 2 release-protected advisory caps enabled.
- Left acceptance fallback disabled.

## Validation

- Imported `get_polymer_notebook_defaults("matrix")` and `get_polymer_notebook_defaults("structured_matrix")` in `rl-env`.
- Confirmed both families now resolve to `Step 3D (Step 2 + usefulness gate)`.
- Ran `py_compile` on `systems/polymer/notebook_params.py`.
