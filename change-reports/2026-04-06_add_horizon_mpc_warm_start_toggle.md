## Summary

Added an explicit warm-start toggle for the shared horizon MPC solve so unified horizon notebooks can switch between:

- shifted previous-solution warm start
- legacy zero-initialized MPC solve at every step

## Files Changed

- `utils/horizon_runner.py`
- `RL_assisted_MPC_horizons_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_unified.ipynb`

## Details

- New horizon config key:
  - `use_shifted_mpc_warm_start`
- Shared horizon runner behavior:
  - `True`: reuse and shift the previous optimal sequence
  - `False`: rebuild a zero initial guess at every MPC step
- Notebook defaults:
  - polymer horizon unified: `True`
  - distillation horizon unified: `False` to match the archived legacy horizon notebook behavior

## Why

This makes it possible to compare the unified distillation horizon notebook against the archived legacy implementation without the MPC warm-start optimization changing the controller trajectory.
