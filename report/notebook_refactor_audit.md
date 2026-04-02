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
- notebook-local supervisor functions
  - `run_dqn_mpc_horizon_supervisor` appears in the two horizon notebooks
  - `run_rl_train_disturbance_gradually` appears in 12 TD3/SAC/residual/weight notebooks
  - the horizon supervisor and the matrix-multiplier supervisor are now migrated

## First migration boundary

- New shared runtime path:
  - `utils/horizon_runner.py`
  - `utils/matrix_runner.py`
  - `utils/rewards.py`
  - `utils/observer.py`
  - `utils/plotting_core.py`
- New notebook entrypoint:
  - `RL_assisted_MPC_horizons_unified.ipynb`
  - `RL_assisted_MPC_matrices_unified.ipynb`
- Frozen references for validation:
  - `RL_assisted_MPC_horizons.ipynb`
  - `RL_assisted_MPC_horizons_dist.ipynb`
  - `RL_assisted_MPC_matrices.ipynb`
  - `RL_assisted_MPC_matrices_dist.ipynb`
  - `RL_assisted_MPC_matrices_SAC.ipynb`
  - `RL_assisted_MPC_matrices_dsit_SAC.ipynb`

## Plotting direction

- `utils/plotting.py` remains the notebook-facing import surface.
- The new horizon and matrix plots now delegate to `utils/plotting_core.py`.
- The TD3-named matrix plotting entrypoint remains available through `utils/plotting.py` as a compatibility wrapper.
- `BasicFunctions/plot_fns.py` should be treated as legacy and should not receive new work.

## Next migration targets

- move the weight supervisor family out of notebooks into a shared runner module
- migrate residual and combined supervisors onto the same normalized result bundle shape
- retire the typo-named `RL_assisted_MPC_matrices_dsit_SAC.ipynb` as a frozen legacy reference rather than a canonical entrypoint
