# Residual Safe-Start Stack With Shadow Safety Logs

Date: 2026-05-29

## Objective

Restore a residual-specific safe-start stack for the active distillation residual default without reintroducing rho authority into the actor state or live execution. The goal is to protect the first TD3 release window while preserving the strong late no-rho residual behavior observed in the latest runs.

## Active Safety Behavior

- Rho remains analysis-only:
  - `append_rho_to_state = False`
  - `authority_use_rho = False`
  - `residual_authority_enabled = False`
- Warm start executes zero residual plant authority.
- Warm-start behavioral cloning trains TD3 toward zero residual / nominal MPC trim only during the warm window.
- Post-warm BC imitation stops, and a 10-subepisode handoff blends zero residual to TD3 authority from `0.1` to `1.0`.
- TD3 residual trust-region caps the scaled residual candidate from `0.005` to `0.02` over 30 post-warm subepisodes.
- Reward probation compares post-warm subepisode reward against the recent warm-start reference. A collapse greater than `5.0` triggers a 2-subepisode cooldown with residual cap `0.005`.
- Nominal emergency fallback executes zero residual only for nonfinite selected/projected residual actions.
- Physical input headroom projection remains active for every residual step.

## Shadow Diagnostics

The runner now saves log-only diagnostics for safety ideas that are not allowed to control the plant in this default:

- shadow rho-authority projection and deadband activity
- diagnostic-only BC release gate pass/block logs
- requested residual, post-handoff residual, post-cap residual, and executed residual
- active residual cap and cap-projection flags
- reward-probation active/trigger logs
- predicted direction/risk placeholder logs as `NaN` fields for later model-based implementation

## Files Changed

- `systems/distillation/notebook_params.py`
- `utils/residual_runner.py`

## Validation

- `python -m py_compile systems/distillation/notebook_params.py utils/residual_runner.py utils/residual_authority.py distillation_RL_assisted_MPC_residual_unified.py`
- No-Aspen config smoke check for residual defaults:
  - rho inactive
  - warm-start BC active
  - post-warm handoff starts at authority `0.1`
  - residual cap ramp active
  - reward probation active
  - release gate diagnostic-only
- Unit-style helper smoke check:
  - residual cap clips scaled `delta_u_res`
  - shadow rho projection returns finite diagnostics
  - active no-rho projection remains independent of shadow rho logs

## Next Run Interpretation

The next residual run should be judged by two separate metrics:

- early release safety: worst first-20 subepisode reward and zero-fallback/projection frequency
- preserved tail authority: late TD3 fraction, residual action magnitude, and final reward compared with the May 28 no-rho residual result

This change should reduce the first-release crash severity without compressing the late residual policy into a nominal-copy regime.
