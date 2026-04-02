# Fix Polymer Notebook Poles Reference

## What changed

Fixed undefined `poles` references in the polymer unified notebooks that pass observer poles into `runtime_ctx`:

- `RL_assisted_MPC_matrices_unified.ipynb`
- `RL_assisted_MPC_weights_unified.ipynb`
- `RL_assisted_MPC_residual_unified.ipynb`
- `RL_assisted_MPC_combined_unified.ipynb`

## Why

These notebooks imported `POLYMER_OBSERVER_POLES` but still passed a bare `poles` variable into `runtime_ctx`, which caused a `NameError` when the cell was run.

## Fix

Each affected notebook now passes:

- `POLYMER_OBSERVER_POLES.copy()`

directly into `runtime_ctx["poles"]`.

## Validation

- `nbformat` read check passed for all active unified notebooks
- verified the four edited polymer notebooks no longer contain `"poles": poles`
