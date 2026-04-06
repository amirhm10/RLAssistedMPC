## Summary

This change updates only the active distillation horizon unified notebook so its RL setpoint scenario is converted into scaled deviation form exactly like the archived `RL_assisted_MPC_horizons.ipynb`.

## What changed

- In `distillation_RL_assisted_MPC_horizons_unified.ipynb`, the RL setpoint pair:
  - `[0.013, -23.0]`
  - `[0.028, -21.0]`
  is now converted as:

`apply_min_max(y_sp_scenario_phys, ...) - apply_min_max(steady_states["y_ss"], ...)`

instead of only:

`apply_min_max(y_sp_scenario_phys, ...)`

## Why

The archived distillation horizon notebook uses setpoints in scaled deviation form. The unified notebook was feeding absolute scaled setpoints into the runner, which changes:

- the MPC tracking target
- the RL state
- the output error `delta_y`
- the reward signal

## Scope

- Only `distillation_RL_assisted_MPC_horizons_unified.ipynb` was changed.
- Reward defaults, observer initialization, MPC warm-start behavior, and other distillation notebooks were left unchanged in this pass.

## Validation

- `nbformat` validation passed.
- All code cells in the edited notebook parse successfully.
