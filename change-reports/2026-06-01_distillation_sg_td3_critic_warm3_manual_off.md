# Distillation SG-TD3 Residual Critic-Warm-3 Manual-Off Entrypoint

## Summary

- Added opt-in `sg_td3` support to the canonical distillation residual entrypoint.
- Added a dedicated distillation SG-TD3 residual critic-warm-3 runner.
- Added distillation residual `sg_td3` run profiles for nominal, ramp disturbance, and fluctuation disturbance.
- Added conservative supervisor-gate defaults to distillation residual config.
- Added tests that validate the distillation SG-TD3 profile and manual-off critic-warm configuration.

## Scientific Guardrails

- Existing TD3, SAC, and TD7 distillation residual entrypoints remain available.
- The new runner uses zero residual as the supervisor action.
- Behavioral cloning, BC handoff, release gate, TD3 authority ramp, rho authority, residual deadband, shadow diagnostics, reward probation, and early-release guard are disabled for this ablation.
- Nonfinite fallback to zero remains enabled.
- Physical input headroom projection remains in the shared runner because it enforces plant input bounds.

## Validation

- Python compile checks were run for the canonical distillation residual entrypoint and the new SG-TD3 critic-warm runner.
- Supervisor-gated TD3 unit tests and residual integration tests were run.
- The full Aspen distillation experiment was not run.
