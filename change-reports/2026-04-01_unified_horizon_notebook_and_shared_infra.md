# 2026-04-01 Unified Horizon Notebook And Shared Infrastructure

## Summary

- Added `RL_assisted_MPC_horizons_unified.ipynb` as the new horizon entrypoint with a single `RUN_MODE` switch for nominal and disturbance runs.
- Added shared modules for reward construction, observer gain, and the horizon supervisor:
  - `utils/rewards.py`
  - `utils/observer.py`
  - `utils/horizon_runner.py`
- Added `utils/plotting_core.py` and wired the horizon plotting/comparison surface in `utils/plotting.py` through the new shared core.
- Added `report/notebook_refactor_audit.md` to record the first notebook migration targets and the current duplication boundaries.

## Behavior Changes

- Horizon experiments can now be launched from one notebook without editing hard-coded absolute paths.
- The disturbance compare path is explicit through `compare_mode=\"disturb\"` in the unified notebook instead of reusing the nominal helper by accident.
- Saved horizon result bundles now carry a normalized `y/u` shape alongside the legacy aliases used by older comparison flows.

## Boundaries Kept In This Pass

- `RL_assisted_MPC_horizons.ipynb` and `RL_assisted_MPC_horizons_dist.ipynb` were left unchanged as reference baselines.
- TD3/SAC/residual/combined notebook supervisors were not migrated out of notebooks yet.
- Legacy plotting paths for weights, multipliers, and multi-agent runs remain available through `utils/plotting.py`.
