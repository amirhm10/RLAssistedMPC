# Unified Residual Notebook And Shared Runtime

## Summary

- Added a unified residual notebook entrypoint for TD3/SAC and nominal/disturbance runs.
- Moved the residual-correction rollout logic into a shared `utils/residual_runner.py` module.
- Extended the shared plotting layer with residual-specific debug plots and normalized residual result-bundle fields.

## Main Changes

- Added `RL_assisted_MPC_residual_unified.ipynb` with:
  - `AGENT_KIND = "td3" | "sac"`
  - `RUN_MODE = "nominal" | "disturb"`
  - repo-relative baseline paths
  - shared reward and observer setup
- Added `run_residual_supervisor(...)` to `utils/residual_runner.py`.
- Added `plot_residual_results(...)` to `utils/plotting.py` and the corresponding plotting-core implementation.
- Updated `report/notebook_refactor_audit.md` to mark the residual correction path as migrated.

## Residual Method Decisions

- The new residual path is correction-only: RL adds a bounded correction to the MPC action.
- Residual state uses the same shared structure as the unified continuous-action notebooks.
- Mismatch-specific RL state augmentation was removed:
  - no innovation term in the RL state
  - no tracking-error term appended to the RL state
  - no mismatch schedule in the unified runtime
- Residual safety is clip-only:
  - residual correction bounds from `low_coef/high_coef`
  - actuator headroom clipping around the MPC baseline action
- Legacy residual mismatch notebooks remain frozen references and were not edited.
- `RL_assisted_MPC_residual_model_mismatch_multi.ipynb` remains out of scope as a separate legacy combined method.

## Validation

- Verified imports for the new runner and residual plotting wrapper in `rl-env`.
- Planned smoke coverage targets:
  - TD3 nominal residual
  - TD3 disturbance residual
  - SAC nominal residual
  - SAC disturbance residual
- Planned compatibility checks:
  - normalized residual result bundle shape
  - no mismatch terms in the unified residual state
  - baseline comparisons use nominal/disturb pickles by mode
