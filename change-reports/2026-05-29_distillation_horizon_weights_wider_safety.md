# Distillation Horizon, Dueling, And Weights Wider Safety Stack

Date: 2026-05-29

## Objective

Widen the active single-runner distillation horizon and weights search spaces while restoring method-specific safe-start behavior. The goal is wider exploration without turning the next batch into nominal-copy behavior or untracked action saturation.

## Active Defaults Changed

- Standard horizon and dueling horizon now use `Np = 2..24` and `Nc = 1..16`.
- The existing recipe builder still filters invalid pairs with `Hc <= Hp`, giving 263 valid horizon actions.
- TD3 weights now uses multiplier bounds `[0.5, 2.5]` for all four penalty multipliers.
- Combined-runner defaults remain on the previous global horizon and weight bounds.

## Safety Behavior Added

- Horizon and dueling:
  - warm start remains default `(6, 3)`
  - post-warm release filter projects early requests into bounded horizon windows
  - reward probation cools down to `(6, 3)` after reward collapse
  - replay stores the executed horizon action
  - selected-vs-default MPC shadow diagnostics are saved every 4 steps
- Weights:
  - warm start executes identity multipliers `[1, 1, 1, 1]`
  - BC trains toward identity only during warm start
  - post-warm handoff blends identity to TD3 over 10 subepisodes
  - multiplier cap ramps from `0.10` to `1.50` around identity over 30 post-warm subepisodes
  - reward probation applies a `0.10` cooldown cap
  - identity fallback handles nonfinite actions and failed selected MPC solves
  - selected-vs-identity MPC shadow diagnostics are saved every 5 steps

## Files Changed

- `systems/distillation/notebook_params.py`
- `utils/horizon_safety.py`
- `utils/horizon_runner.py`
- `utils/horizon_runner_dueling.py`
- `utils/weights_runner.py`
- `distillation_RL_assisted_MPC_horizons_unified.py`
- `distillation_RL_assisted_MPC_horizons_dueling_unified.py`
- `distillation_RL_assisted_MPC_weights_unified.py`

## Validation

- Static validation passed with `py_compile` for the changed runners and unified entrypoints.
- No-Aspen horizon smoke check passed:
  - standard and dueling each build 263 valid recipes
  - `(6, 3)` is present
  - aggressive first post-warm request projects into protected window
  - cooldown forces `(6, 3)`
- No-Aspen weights smoke check passed:
  - bounds are `[0.5, 2.5]`
  - warm-start BC is active
  - handoff starts after warm start at authority `0.1`
  - release gate is diagnostic-only
  - cooldown cap overrides normal cap

## Next Run Interpretation

The next run should be analyzed as a safety-restored wider-search experiment:

- horizon success means wider search improves reward without high projection/cooldown frequency or excessive horizon switching
- dueling success means the concentrated policy remains stable despite the 263-action space
- weights success means tail reward remains high while saturation and identity fallback stay low
