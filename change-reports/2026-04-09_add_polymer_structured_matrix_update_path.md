## Summary

Added a new polymer-only structured model-update path for RL-assisted MPC without changing the existing scalar matrix notebook or runner.

## What Was Added

- `utils/structured_model_update.py`
  - split augmented physical/disturbance blocks
  - build block-lite structured updates
  - build band-lite structured updates
  - map normalized RL actions to positive multiplier ranges
  - validate preserved disturbance-block structure
- `utils/structured_matrix_runner.py`
  - new structured matrix supervisor runner
  - same rollout/reward/replay/MPC flow as the current matrix path
  - structured diagnostics and saved metadata
- `RL_assisted_MPC_structured_matrices_unified.ipynb`
  - new polymer-only notebook
  - mirrors the current polymer matrix notebook structure
  - supports `UPDATE_FAMILY = "block" | "band"`
- polymer notebook defaults in `systems/polymer/notebook_params.py`
  - new `structured_matrix` family
  - tight range profile as the default
- plotting additions in `utils/plotting.py` / `utils/plotting_core.py`
  - new structured matrix plotting wrapper
  - appends structured multiplier/model-delta diagnostics while keeping the existing matrix plotting path unchanged

## Defaults Chosen

- primary default experiment: block-lite structured scaling
- polymer-only notebook in this pass
- `UPDATE_FAMILY = "block"`
- `RANGE_PROFILE = "tight"`
- `BLOCK_GROUP_COUNT = 3`
- `BLOCK_GROUPS = None` for contiguous equal-size physical-state groups
- `BAND_OFFSETS = [0, 1, 2]`
- `RECALCULATE_OBSERVER_ON_MATRIX_CHANGE = False`

## Validation

- `python -m py_compile` passed for:
  - `utils/structured_model_update.py`
  - `utils/structured_matrix_runner.py`
  - `utils/plotting.py`
  - `utils/plotting_core.py`
  - `systems/polymer/notebook_params.py`
- parsed `RL_assisted_MPC_structured_matrices_unified.ipynb` with `nbformat`/`ast`
- helper smoke checks confirmed:
  - block-lite and band-lite identity multipliers reproduce the nominal augmented model
  - block and band action dimensions match the configured layouts
- short `rl-env` smoke run confirmed:
  - the new structured runner executes end-to-end in polymer block mode
  - `plot_structured_matrix_results(...)` saves output successfully

## Preserved Behavior

- `RL_assisted_MPC_matrices_unified.ipynb` remains unchanged
- `utils/matrix_runner.py` remains unchanged
- polymer system models, reward design, replay logic, and MPC objective were not changed for the existing path
