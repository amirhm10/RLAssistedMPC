# Horizon Reward Scaling Fix

## Summary
- Removed the extra default reward down-scaling from the shared horizon runner.
- Updated the polymer and distillation unified horizon notebooks to pass `reward_scale = 1.0`.

## Why
- `utils.rewards.make_reward_fn_relative_QR(...)` already applies the final `* 0.01` scaling inside the shared reward function.
- The horizon path still carried an older notebook-era `reward_scale = 0.01` hook, which would have scaled horizon rewards a second time if those notebooks were re-run.
- The other unified continuous-action runners already use the shared reward directly without that extra factor.

## Files
- `utils/horizon_runner.py`
- `RL_assisted_MPC_horizons_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_unified.ipynb`
