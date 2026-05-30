# Raise Distillation Temperature Reward Weight

Date: 2026-05-30

## Summary

Raised the shared distillation reward temperature penalty so active distillation RL runners are more strongly penalized for tray-85 temperature tracking error.

## Change

- Updated `systems/distillation/config.py`.
- Changed `RL_REWARD_DEFAULTS["Q_diag"]` from `[3.7e4, 5.0e3]` to `[3.7e4, 2.0e4]`.

## Rationale

Recent five-runner analysis showed several runners improve tray-24 composition while remaining worse than OF-MPC on tray-85 temperature. In scaled reward-band terms, the old temperature edge contribution was much smaller than the composition contribution. The new value increases temperature pressure while keeping composition slightly dominant.

## Validation

- Compile the distillation config and notebook defaults.
- Smoke check that active distillation notebook defaults now copy the new `Q_diag`.
- No full Aspen simulation was run.
