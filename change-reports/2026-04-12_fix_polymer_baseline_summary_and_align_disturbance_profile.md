## Summary

Fixed the polymer offset-free MPC notebook summary crash and aligned the
polymer baseline disturbance defaults with the active RL polymer notebooks.

## Changes

- `utils/notebook_setup.py`
  - `print_grouped_notebook_summary(...)` now accepts both:
    - grouped dictionaries of dictionaries
    - flat summary dictionaries
  - this fixes the crash in `MPCOffsetFree_unified.ipynb`, where the notebook
    passed a flat summary block

- `systems/polymer/notebook_params.py`
  - aligned the polymer baseline disturbance profile with the active RL
    polymer defaults
  - baseline disturbance defaults now use:
    - `qi_change = 0.85`
    - `qs_change = 1.3`
    - `ha_change = 0.85`

## Validation

- `py_compile` passed for:
  - `utils/notebook_setup.py`
  - `systems/polymer/notebook_params.py`
- verified `print_grouped_notebook_summary(...)` works for both flat and grouped
  inputs
- verified polymer baseline disturbance defaults now match the polymer horizon
  disturbance defaults
