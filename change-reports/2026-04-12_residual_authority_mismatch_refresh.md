# Residual Authority Refresh Plus Repo-Wide Mismatch V2

## Scope

Implemented the mismatch-v2 refresh across the active RL notebook families that use the shared RL state builder, and moved residual authority/projection into one shared residual-only helper used by both standalone residual and the combined residual branch.

Covered notebook families:

- Polymer: horizon, dueling horizon, matrix, structured matrix, weights, residual, combined, reid_batch, reid_batch_v2
- Distillation: horizon, dueling horizon, matrix, structured matrix, weights, residual, combined

Out of scope:

- baseline notebooks
- system-identification notebooks

## Code Changes

- Replaced the old shared mismatch-scale path in [utils/state_features.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\state_features.py) with reward-band-based mismatch-v2 helpers.
- Added [utils/residual_authority.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\residual_authority.py) as the single source of truth for residual projection and authority handling.
- Updated these runners to use mismatch-v2:
  - [utils/horizon_runner.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\horizon_runner.py)
  - [utils/horizon_runner_dueling.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\horizon_runner_dueling.py)
  - [utils/matrix_runner.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\matrix_runner.py)
  - [utils/structured_matrix_runner.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\structured_matrix_runner.py)
  - [utils/weights_runner.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\weights_runner.py)
  - [utils/residual_runner.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\residual_runner.py)
  - [utils/combined_runner.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\combined_runner.py)
  - [utils/reid_batch_runner.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\reid_batch_runner.py)
- Extended [utils/plotting_core.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\plotting_core.py) to normalize and plot the new residual authority diagnostics while staying backward-compatible with older bundles.

## Notebook / Defaults Changes

- Added mismatch-v2 notebook-facing defaults to:
  - [systems/polymer/notebook_params.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\systems\polymer\notebook_params.py)
  - [systems/distillation/notebook_params.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\systems\distillation\notebook_params.py)
- Added residual-only authority defaults to the residual and combined families in both systems.
- Updated the active unified notebooks so they:
  - stop using `default_mismatch_scale(...)`
  - expose the new mismatch settings
  - append `rho` to residual state dimensions only when mismatch mode is active and `append_rho_to_state=True`
  - print mismatch settings in the grouped summaries
  - print residual authority settings in residual and combined summaries

## Validation

- `py_compile` passed for all touched Python modules.
- Parsed all touched unified notebooks with `nbformat` + `ast`.
- Ran a small `rl-env` helper smoke check covering:
  - mismatch-v2 config resolution
  - mismatch-v2 state construction with residual rho append
  - residual authority projection helper output shape and execution path
