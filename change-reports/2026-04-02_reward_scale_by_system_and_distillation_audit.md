# Reward Scale By System And Distillation Audit

## What changed

- `utils/rewards.py`
  - `make_reward_fn_relative_QR(...)` now accepts `reward_scale`
  - the final reward scaling is now explicit and system-controlled
- `systems/polymer/config.py`
  - `RL_REWARD_DEFAULTS["reward_scale"] = 0.01`
- `systems/distillation/config.py`
  - `RL_REWARD_DEFAULTS["reward_scale"] = 1.0`
- `utils/horizon_runner.py`
  - removed the extra runner-side reward scaling
- `utils/mpc_baseline_runner.py`
  - removed the extra runner-side reward scaling
- `RL_assisted_MPC_horizons_unified.ipynb`
  - removed stale `reward_scale` from the runtime config dict
- `distillation_RL_assisted_MPC_horizons_unified.ipynb`
  - removed stale `reward_scale` from the runtime config dict

## Why

The archived distillation RL notebooks use the same relative-band reward form as the unified notebooks, but they return the raw reward without a final `* 0.01`.

The shared helper had a fixed `* 0.01`, which compressed distillation reward ranges relative to the archived notebooks.

This change makes reward scaling an explicit system-level parameter:

- polymer keeps the historical `0.01`
- distillation uses `1.0`

## Audit result

The archived distillation notebook families split into two groups:

- `RL_assisted_MPC_horizons*`, `RL_assisted_MPC_matrices*`, `RL_assisted_MPC_weights*`
  - use inline `make_reward_fn_relative_QR(...)`
  - unscaled final reward
- `MPCOffsetFree*` and `RL_assisted_MPC_combined.ipynb`
  - use `make_reward_fn_fixed_QR(...)`
  - also unscaled, but with different reward parameters

That means the new reward-scale fix aligns the unified distillation RL notebook family that uses the relative-band reward, but does not by itself reproduce the archived distillation baseline or combined reward ranges.
