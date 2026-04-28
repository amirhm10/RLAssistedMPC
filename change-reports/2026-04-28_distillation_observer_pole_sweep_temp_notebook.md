# 2026-04-28 Distillation observer pole sweep temporary notebook

## Summary

Added a temporary notebook:

- `distillation_MPCOffsetFree_observer_pole_sweep_temp.ipynb`

The notebook is a nominal distillation offset-free MPC observer-pole sweep. It is notebook-local and does not modify shared distillation defaults.

## Behavior

- Forces `run_mode = "nominal"` and `n_tests = 2`
- Loads the steady-state/scaling context once
- For each observer-pole candidate:
  - opens a fresh Aspen session
  - builds the observer gain `L`
  - runs the offset-free MPC baseline
  - saves a per-candidate bundle
  - optionally saves per-run baseline plots
  - closes Aspen before moving to the next candidate
- Writes a summary CSV, a manifest JSON, and sweep-comparison plots at the end

## Outputs

The notebook writes sweep artifacts under temporary subdirectories inside the canonical distillation data/result roots:

- `Distillation/Data/observer_pole_sweep_temp/<timestamp>/`
- `Distillation/Results/observer_pole_sweep_temp/<timestamp>/`

## Validation

- Notebook JSON validated with `nbformat`
- All code cells parsed successfully with Python `ast`
