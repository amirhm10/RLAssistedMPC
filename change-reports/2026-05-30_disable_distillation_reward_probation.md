# Disable Distillation Reward Probation

Date: 2026-05-30

## Summary

Disabled reward-collapse probation for the active five single-runner distillation defaults while keeping the other safety layers active.

## Changed Behavior

- Markov keeps TD3-priority fallback, authority ramp, LS fallback, and z-safety, but no longer applies reward-probation cooldown scaling.
- Weights keeps identity warm start, BC handoff, multiplier cap ramp, identity fallback, and shadow identity diagnostics, but no reward-probation cooldown cap.
- Standard and dueling horizon keep the widened horizon grid, release filter, default fallback, and shadow default-MPC diagnostics, but no reward-probation cooldown to `(6, 3)`.
- Residual keeps warm zero, BC handoff, residual cap ramp, zero fallback, and shadow safety configs, but no reward-probation cooldown cap.

## Validation Plan

- `py_compile` the changed distillation defaults and the five unified entrypoints.
- Load active defaults without Aspen and assert all five reward-probation flags are disabled.
- Confirm non-probation safety layers remain enabled.
