# Unified Weights Notebook And TD3 Evaluation Fix

## Summary

- Added `RL_assisted_MPC_weights_unified.ipynb` as the new MPC penalty-multiplier entrypoint.
- Added `utils/weights_runner.py` to move the shared TD3/SAC penalty-multiplier supervisor out of notebook-local cells.
- Extended the shared plotting layer with a weight-specific result path and kept the legacy TD3-named weight plotter as a compatibility wrapper.
- Updated both TD3 agent implementations so evaluation uses the live actor with exploration off instead of the target actor.

## Files Added

- `RL_assisted_MPC_weights_unified.ipynb`
- `utils/weights_runner.py`
- `change-reports/2026-04-01_unified_weights_notebook_and_td3_eval_fix.md`

## Files Updated

- `TD3Agent/agent.py`
- `TD3Agent/agent_modified.py`
- `utils/plotting.py`
- `utils/plotting_core.py`

## Behavior Changes

- The unified weights notebook switches experiments with two top-level controls:
  - `AGENT_KIND = "td3" | "sac"`
  - `RUN_MODE = "nominal" | "disturb"`
- Baseline MPC comparison is normalized by mode:
  - nominal -> `Data/mpc_results_nominal.pickle`
  - disturbance -> `Data/mpc_results_dist.pickle`
- The shared weights runner returns a normalized result bundle with:
  - rollout data (`y`, `u`, `avg_rewards`, `delta_y_storage`, `delta_u_storage`)
  - penalty logs (`weight_log`, `low_coef`, `high_coef`)
  - train/test metadata and disturbance profile when present
  - agent training diagnostics (`actor_losses`, `critic_losses`, and SAC alpha traces when available)
- Weight warm-start uses identity multipliers `[1, 1, 1, 1]` without calling the agent to infer action shape.
- TD3 `act_eval(...)` now uses the live actor in both `TD3Agent/agent.py` and `TD3Agent/agent_modified.py`.

## Boundaries

- Left the two legacy TD3 weights notebooks unchanged:
  - `RL_assisted_MPC_weights.ipynb`
  - `RL_assisted_MPC_weights_dist.ipynb`
- Left all model-mismatch notebooks untouched and excluded mismatch logic from the new weights path.
- Did not stage unrelated dirty notebooks, result folders, or generated data already present in the worktree.
