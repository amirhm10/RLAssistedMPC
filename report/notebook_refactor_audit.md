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
  - only the horizon supervisor is migrated in this pass

## First migration boundary

- New shared runtime path:
  - `utils/horizon_runner.py`
  - `utils/rewards.py`
  - `utils/observer.py`
  - `utils/plotting_core.py`
- New notebook entrypoint:
  - `RL_assisted_MPC_horizons_unified.ipynb`
- Frozen references for validation:
  - `RL_assisted_MPC_horizons.ipynb`
  - `RL_assisted_MPC_horizons_dist.ipynb`

## Plotting direction

- `utils/plotting.py` remains the notebook-facing import surface.
- The new horizon plots now delegate to `utils/plotting_core.py`.
- The TD3/weight/multi-agent plotting functions remain available through `utils/plotting.py` and are not migrated yet.
- `BasicFunctions/plot_fns.py` should be treated as legacy and should not receive new work.

## Next migration targets

- move the TD3/weight supervisor family out of notebooks into a shared runner module
- unify baseline MPC comparison calls so disturbance notebooks stop pointing at nominal comparison helpers
- normalize saved result payloads for matrices, weights, residual, and combined notebooks onto the same result bundle shape
