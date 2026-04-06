## Summary

Aligned the shared distillation RL reward default with the archived working distillation horizon notebook.

## Files Changed

- `systems/distillation/config.py`

## What Changed

- Updated `RL_REWARD_DEFAULTS["beta"]`:
  - before: `5.0`
  - after: `7.0`

## Why

The archived working notebook `DIstillation Column Case/RL_assisted_MPC_DL/RL_assisted_MPC_horizons.ipynb` uses `beta=7.0` in the actual reward-function call. All active distillation unified runtime notebooks read their default reward configuration from `systems/distillation/config.py`, so updating this single value propagates the legacy-aligned default consistently across:

- baseline MPC
- horizon
- matrix multipliers
- weight multipliers
- residual correction
- combined supervisor
