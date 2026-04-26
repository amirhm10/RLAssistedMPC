# Shared Combined Agent Logic And Dueling Horizon Choice

## Summary

The combined polymer and distillation supervisors now use shared step-level helpers for horizon, matrix, weights, and residual action selection, replay storage, and training gates. This keeps the combined agent behavior aligned with the single-agent runners while leaving the combined runner responsible for the multi-agent MPC rollout.

## Changes

- Added `utils/agent_step_runtime.py` for common discrete horizon and continuous TD3/SAC step behavior.
- Refactored standard horizon, dueling horizon, matrix, weights, residual, and combined runners to call the shared helpers.
- Added combined `horizon_agent_kind` support with `"dqn"` and `"dueling_dqn"` options.
- Added `horizon_dueling_agent` defaults to polymer and distillation combined defaults.
- Updated combined notebooks to expose `HORIZON_AGENT_KIND`, instantiate `DQNAgent` or `DuelingDQNAgent`, and pass the selected kind into `combined_cfg`.
- Preserved the scalar combined matrix block with Step 2 release-protected advisory caps and no Step 3 fallback.

## Validation

- Notebook JSON validation and Python compilation should be run after this change.
- Smoke checks should verify both standard and dueling horizon selection paths without executing the full long polymer or distillation workflows.
