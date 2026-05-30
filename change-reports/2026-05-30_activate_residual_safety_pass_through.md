# Activate Residual Safety Pass-Through

Date: 2026-05-30

## Summary

The active distillation residual defaults already contained `residual_safety`, but the unified residual entrypoint did not pass that block into `run_residual_supervisor`. This made saved residual bundles report `residual_safety_enabled=False`.

## Change

- Added `RESIDUAL_SAFETY_CFG = dict(NB.get("residual_safety", {}))`.
- Added `"residual_safety": dict(RESIDUAL_SAFETY_CFG)` to `residual_cfg`.

## Expected Behavior

Future residual runs should save `residual_safety_enabled=True` and include the configured residual safety block. Reward probation remains disabled by the latest defaults, so this activates non-probation residual safety fields only.
