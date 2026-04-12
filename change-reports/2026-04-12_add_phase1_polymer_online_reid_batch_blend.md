## Summary

Added a new polymer-only unified workflow for online batch re-identification of
the physical prediction model with RL control over a scalar blend factor
between the nominal and identified models.

## What Was Added

- `utils/reid_batch.py`
  - rolling physical-state identification buffer
  - polymer phase-1 scalar matrix basis
  - batch regression assembly
  - ridge and bounded least-squares solvers
  - model reconstruction, candidate selection, eta mapping, and blend-state helpers
- `utils/reid_batch_runner.py`
  - new `run_reid_batch_supervisor(...)` entrypoint
  - fixed nominal observer with online blended prediction model
  - online identification updates and RL blend control
- `RL_assisted_MPC_reid_batch_unified.ipynb`
  - new polymer unified notebook
  - mirrors the current matrix-notebook structure
  - supports TD3 and SAC with a scalar action
- `systems/polymer/notebook_params.py`
  - new `reid_batch` default family
- `utils/plotting.py` / `utils/plotting_core.py`
  - new `plot_reid_batch_results(...)` plotting entrypoint
  - extra eta, theta, ID residual, and model-delta diagnostics
- `report/05_online_reidentification_batch_ridge_blend.tex`
  - method note for the new workflow

## Runtime Semantics

- observer stays nominal and fixed
- MPC prediction model is blended online:
  - nominal when `eta = 0`
  - identified when `eta = 1`
- RL action is one scalar raw action mapped to `eta_raw in [0, 1]`, then smoothed
- identification updates run on the physical state block only
- the ID layer owns candidate acceptance/fallback to previous valid or nominal model

## Defaults Chosen

- polymer only in phase 1
- default agent: `td3`
- default state mode: `mismatch`
- default ID solver: `ridge_closed_form`
- `id_window = 80`
- `id_update_period = 5`
- `lambda_prev = 1e-2`
- `lambda_0 = 1e-4`
- `eta_smoothing_tau = 0.1`

## Validation

- `py_compile` passed for:
  - `utils/reid_batch.py`
  - `utils/reid_batch_runner.py`
  - `utils/plotting.py`
  - `utils/plotting_core.py`
  - `systems/polymer/notebook_params.py`
- parsed `RL_assisted_MPC_reid_batch_unified.ipynb` with `ast`
- `rl-env` synthetic identification recovery check passed for the phase-1 polymer basis
- `rl-env` smoke run passed with:
  - `force_eta_constant = 0.0`
  - `disable_identification = True`
  - zero blended-model deviation, confirming the nominal invariance path
- `rl-env` plotting smoke passed for `plot_reid_batch_results(...)`
- `rl-env` active-identification smoke passed with a forced nonzero eta path

## Preserved Behavior

- no existing unified notebook or runner was changed behaviorally
- no work was routed through `Simulation/rl_sim.py`
- no distillation re-identification path was added in phase 1
