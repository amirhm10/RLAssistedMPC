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

## First migration boundary

- New shared runtime path:
  - `utils/horizon_runner.py`
  - `utils/combined_runner.py`
  - `utils/matrix_runner.py`
  - `utils/residual_runner.py`
  - `utils/weights_runner.py`
  - `utils/rewards.py`
  - `utils/observer.py`
  - `utils/state_features.py`
  - `utils/plotting_core.py`
- New notebook entrypoint:
  - `RL_assisted_MPC_combined_unified.ipynb`
  - `RL_assisted_MPC_horizons_unified.ipynb`
  - `RL_assisted_MPC_matrices_unified.ipynb`
  - `RL_assisted_MPC_residual_unified.ipynb`
  - `RL_assisted_MPC_weights_unified.ipynb`
- Frozen references for validation:
  - `RL_assisted_MPC_horizons.ipynb`
  - `RL_assisted_MPC_horizons_dist.ipynb`
  - `RL_assisted_MPC_matrices.ipynb`
  - `RL_assisted_MPC_matrices_dist.ipynb`
  - `RL_assisted_MPC_matrices_SAC.ipynb`
  - `RL_assisted_MPC_matrices_dsit_SAC.ipynb`
  - `RL_assisted_MPC_weights.ipynb`
  - `RL_assisted_MPC_weights_dist.ipynb`
  - `RL_assisted_MPC_residual_model_mismatch.ipynb`
  - `RL_assisted_MPC_residual_model_mismatch1.ipynb`
  - `RL_assisted_MPC_residual_model_mismatch2.ipynb`

## Plotting direction

- `utils/plotting.py` remains the notebook-facing import surface.
- The new horizon, matrix, weight, and residual plots now delegate to `utils/plotting_core.py`.
- The TD3-named matrix plotting entrypoint remains available through `utils/plotting.py` as a compatibility wrapper.
- `BasicFunctions/plot_fns.py` should be treated as legacy and should not receive new work.

## Next migration targets

- decide whether the mismatch-state mode should eventually become the default for any unified notebook family, or remain an explicit opt-in switch
- retire the typo-named `RL_assisted_MPC_matrices_dsit_SAC.ipynb` as a frozen legacy reference rather than a canonical entrypoint
- keep `RL_assisted_MPC_residual_model_mismatch_multi.ipynb` as a frozen legacy combined method; it remains out of scope for the residual-correction unification
- decide whether saved-run ablation overlays should be added on top of the new baseline-only combined comparison flow
