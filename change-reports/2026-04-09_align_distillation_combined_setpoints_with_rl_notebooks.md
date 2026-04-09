# Align Distillation Combined Setpoints With RL Notebooks

## Summary

The distillation combined notebook was using a different supervisory setpoint
pair from the other distillation RL notebooks during the warm-start MPC-only
phase:

- combined: `[[0.013, -23.0], [0.018, -22.0]]`
- horizon / matrix / weights / residual: `[[0.013, -23.0], [0.028, -21.0]]`

This made the warm-start rewards look inconsistent even before any RL action
was applied.

## Changes

- `systems/distillation/config.py`
  - changed `DISTILLATION_COMBINED_SETPOINTS_PHYS` to mirror
    `DISTILLATION_RL_SETPOINTS_PHYS`
- `distillation_RL_assisted_MPC_combined_unified.ipynb`
  - switched the combined notebook to read `SYS["rl_setpoints_phys"]`
    explicitly

## Validation

- `python -m py_compile systems/distillation/config.py`
- parsed `distillation_RL_assisted_MPC_combined_unified.ipynb` code cells with `ast`
