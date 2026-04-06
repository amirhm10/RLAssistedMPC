## Summary

Applied the same distillation RL setpoint fix used in the unified horizon notebook to the other distillation unified RL notebooks.

## Files Changed

- `distillation_RL_assisted_MPC_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_weights_unified.ipynb`
- `distillation_RL_assisted_MPC_residual_unified.ipynb`
- `distillation_RL_assisted_MPC_combined_unified.ipynb`

## What Changed

- Changed `y_sp_scenario` construction from absolute scaled outputs to scaled deviation form:
  - before: `apply_min_max(y_sp_scenario_phys, ...)`
  - after: `apply_min_max(y_sp_scenario_phys, ...) - apply_min_max(steady_states["y_ss"], ...)`

## Why

The archived distillation RL notebooks use setpoints in scaled deviation coordinates. The unified RL notebooks were still using absolute scaled coordinates, which changed:

- RL state construction
- tracking error
- reward evaluation
- comparison behavior against baseline runs

## Disturbance Audit

The active distillation unified notebooks already pass:

- `DISTURBANCE_SCHEDULE = build_distillation_disturbance_schedule(...)`
- `system_stepper = distillation_system_stepper`

into the shared runners. That means distillation runs use the distillation-specific disturbance implementation in normal operation. The polymer disturbance builder still exists only as a fallback path if a caller omits the explicit distillation schedule.
