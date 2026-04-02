# Notebook Refactor Audit

## Current duplication targets

- `make_reward_fn_relative_QR`
  - duplicated across 20 notebooks
  - moved to `utils/rewards.py`
- `compute_observer_gain`
  - duplicated across 14 notebooks
  - moved to `utils/observer.py`
- `generate_setpoints_training_rl_gradually`
  - already shared in `utils/helpers.py`
  - should remain the single source for gradual schedule generation
- RL state construction
  - the unified notebooks previously duplicated the standard state-dimension formula
  - `utils/state_features.py` is now the shared home for `get_rl_state_dim(...)` and `build_rl_state(...)`
  - the unified notebooks now expose `STATE_MODE = "standard" | "mismatch"` and use the shared builder
- notebook-local supervisor functions
  - `run_dqn_mpc_horizon_supervisor` appears in the two horizon notebooks
  - `run_rl_train_disturbance_gradually` appears in 12 TD3/SAC/residual/weight notebooks
  - the horizon supervisor, matrix-multiplier supervisor, weight-multiplier supervisor, and residual-correction supervisor are now migrated

## Active unified surface

- New shared runtime path:
  - `utils/mpc_baseline_runner.py`
  - `utils/horizon_runner.py`
  - `utils/combined_runner.py`
  - `utils/matrix_runner.py`
  - `utils/residual_runner.py`
  - `utils/weights_runner.py`
  - `utils/rewards.py`
  - `utils/observer.py`
  - `utils/state_features.py`
  - `utils/plotting_core.py`
  - `systems/distillation/`
- New notebook entrypoint:
  - `MPCOffsetFree_unified.ipynb`
  - `RL_assisted_MPC_combined_unified.ipynb`
  - `RL_assisted_MPC_horizons_unified.ipynb`
  - `RL_assisted_MPC_matrices_unified.ipynb`
  - `RL_assisted_MPC_residual_unified.ipynb`
  - `RL_assisted_MPC_weights_unified.ipynb`
  - `distillation_systemIdentification_unified.ipynb`
  - `distillation_MPCOffsetFree_unified.ipynb`
  - `distillation_RL_assisted_MPC_horizons_unified.ipynb`
  - `distillation_RL_assisted_MPC_matrices_unified.ipynb`
  - `distillation_RL_assisted_MPC_weights_unified.ipynb`
  - `distillation_RL_assisted_MPC_residual_unified.ipynb`
  - `distillation_RL_assisted_MPC_combined_unified.ipynb`

## Distillation migration notes

- The canonical distillation system now lives under `systems/distillation/` with:
  - plant adapter
  - canonical scenario generation
  - canonical data/result naming
  - label metadata for plotting
  - system-identification helpers
- Canonical distillation data now lives under `Data/distillation/`.
- Canonical distillation plot output now lives under `Result/distillation/`.
- The archived subtree `DIstillation Column Case/RL_assisted_MPC_DL/` remains a read-only reference and should not receive new work.

## Removed legacy notebook families

- Split baseline MPC notebooks were removed after `MPCOffsetFree_unified.ipynb` was added.
- Split horizon notebooks were removed after `RL_assisted_MPC_horizons_unified.ipynb` was added.
- Split matrix notebooks, including the typo-named SAC variant, were removed after `RL_assisted_MPC_matrices_unified.ipynb` was added.
- Split weight notebooks were removed after `RL_assisted_MPC_weights_unified.ipynb` was added.
- Legacy residual mismatch notebooks were removed after mismatch-state mode and residual mismatch authority were restored in `RL_assisted_MPC_residual_unified.ipynb`.
- Legacy combined notebooks were removed after `RL_assisted_MPC_combined_unified.ipynb` was added.

## Plotting direction

- `utils/plotting.py` remains the notebook-facing import surface.
- The new horizon, matrix, weight, and residual plots now delegate to `utils/plotting_core.py`.
- The TD3-named matrix plotting entrypoint remains available through `utils/plotting.py` as a compatibility wrapper.
- `BasicFunctions/plot_fns.py` should be treated as legacy and should not receive new work.

## Next migration targets

- decide whether the mismatch-state mode should eventually become the default for any unified notebook family, or remain an explicit opt-in switch
- decide whether saved-run ablation overlays should be added on top of the new baseline-only combined comparison flow
