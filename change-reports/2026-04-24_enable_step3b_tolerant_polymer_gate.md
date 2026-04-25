# Enable Step 3B Tolerant Polymer Gate

Date: 2026-04-24

## Summary

Changed the polymer matrix and structured-matrix Step 3 acceptance defaults from a strict no-worse nominal objective gate to a small nominal-cost trust-region gate.

## Change

- Polymer `mpc_acceptance_fallback.relative_tolerance` is now `1e-4` when Step 3 is enabled.
- Polymer `absolute_tolerance` remains `1e-8`.
- Distillation matrix and structured-matrix defaults remain disabled and unchanged.

## Reason

The strict Step 3 run fell back to nominal MPC for nearly all live decisions. A `1e-4` relative tolerance allows only `0.010%` nominal-cost slack while giving the candidate model enough room to execute when it is close to nominal MPC under the nominal objective.

## Files

- `systems/polymer/notebook_params.py`
- `report/matrix_multiplier_cap_calculation_and_distillation_recovery.md`
