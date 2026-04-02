# Unified Offset-Free MPC Notebook

## Summary

- Added `MPCOffsetFree_unified.ipynb` as the canonical baseline MPC entrypoint with a shared `RUN_MODE = "nominal" | "disturb"` switch.
- Added `utils/mpc_baseline_runner.py` as the shared offset-free MPC runtime.
- Added `plot_baseline_mpc_results(...)` as the public plotting wrapper over the shared plotting core.

## What Changed

- Baseline MPC rollout logic is now centralized in `utils/mpc_baseline_runner.py` instead of staying notebook-local.
- Baseline plotting now uses the shared plotting layer and supports `STYLE_PROFILE = "hybrid" | "paper" | "debug"`.
- The unified notebook saves canonical baseline pickles by mode:
  - `Data/mpc_results_nominal.pickle`
  - `Data/mpc_results_dist.pickle`
- The saved baseline payload remains compatible with existing RL comparison flows by keeping legacy keys such as `u_mpc`, `y_mpc`, and `delat_u_storage`.

## Important Notes

- `MPCOffsetFree.ipynb` and `MPCOffsetFreeDist.ipynb` remain untouched as frozen references.
- `MPCOffsetFreeDist2.ipynb` was not migrated in this pass.
- The shared reward function already includes the historical `0.01` scaling, so the baseline runner now defaults to `reward_scale = 1.0` to avoid double scaling.

## Validation

- Python compile/import checks for:
  - `utils/mpc_baseline_runner.py`
  - `utils/plotting.py`
  - `utils/plotting_core.py`
- Notebook schema validation for `MPCOffsetFree_unified.ipynb`
- Short nominal and disturbance smoke runs in `rl-env`
- Baseline plotting smoke tests for `hybrid`, `paper`, and `debug`
