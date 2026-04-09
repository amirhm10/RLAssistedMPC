## Summary

Added a new distillation structured matrix notebook and defaults entry so the structured-model-update workflow is available for distillation alongside the polymer path, without changing the existing scalar distillation matrix notebook.

## What Was Added

- `distillation_RL_assisted_MPC_structured_matrices_unified.ipynb`
  - mirrors the current distillation matrix notebook structure
  - switches the supervisor path to the shared structured matrix runner
  - exposes:
    - `UPDATE_FAMILY = "block" | "band"`
    - `RANGE_PROFILE = "tight" | "default" | "wide"`
    - `BLOCK_GROUP_COUNT`
    - `BLOCK_GROUPS`
    - `BAND_OFFSETS`
    - `RECALCULATE_OBSERVER_ON_MATRIX_CHANGE`
- `systems/distillation/notebook_params.py`
  - new `structured_matrix` defaults family
  - reuses the current distillation matrix agent/reward/system defaults
  - adds the structured-update controls with block-lite + tight bounds as the first default

## Important Choice

- The new notebook keeps using the existing distillation matrix Aspen-family mapping:
  - `matrix_td3`
  - `matrix_sac`

This avoids introducing a new Aspen family or changing the current distillation plant-path rules.

## Validation

- `python -m py_compile systems/distillation/notebook_params.py`
- parsed `distillation_RL_assisted_MPC_structured_matrices_unified.ipynb` with `nbformat`/`ast`
- short `rl-env` helper smoke check confirmed:
  - block-lite and band-lite specs build from the distillation canonical model bundle
  - identity multipliers reproduce the nominal augmented model

## Preserved Behavior

- `distillation_RL_assisted_MPC_matrices_unified.ipynb` remains unchanged
- no change to the scalar matrix runner, reward logic, replay logic, or distillation plant implementation
