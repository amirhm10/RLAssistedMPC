# 2026-04-28 Step 3C Shadow Diagnostics Implementation

## Summary

Implemented Step 3C as a shadow-only dual-cost diagnostic layer for the polymer scalar and structured matrix supervisors. The shared polymer defaults remain on Step 4G. Step 3C is enabled only through notebook-local study overrides. Distillation receives the same config surface, but it stays disabled by default.

## Code changes

- Added `mpc_dual_cost_shadow` defaults to:
  - `systems/polymer/notebook_params.py`
  - `systems/distillation/notebook_params.py`
- Extended `utils/mpc_acceptance_gate.py` with:
  - `run_mpc_dual_cost_shadow(...)`
  - shadow reason codes for disabled, evaluated, and candidate-solve-failure fallback
- Wired Step 3C logs into:
  - `utils/matrix_runner.py`
  - `utils/structured_matrix_runner.py`

## Notebook changes

- `RL_assisted_MPC_matrices_unified.ipynb`
- `RL_assisted_MPC_structured_matrices_unified.ipynb`

Both polymer notebooks now apply a local Step 3C study override:

- Step 2 on
- Step 4 off
- Step 3B hard fallback off
- Step 3C shadow on

Dedicated study prefixes were also added so these runs do not mix with Step 4G outputs.

## Report update

- Extended `report/matrix_multiplier_cap_calculation_and_distillation_recovery.md`
  with a Step 3C implementation note covering the shadow equations, polymer study setup, and the polymer-on / distillation-off policy.
