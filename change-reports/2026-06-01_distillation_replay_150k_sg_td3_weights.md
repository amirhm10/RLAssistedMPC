# Distillation replay 150k and SG-TD3 weights runner

Date: 2026-06-01

## Summary

- Raised the active distillation replay-buffer default from 50k to 150k transitions.
- Added SG-TD3 support to the distillation weight-multiplier entrypoint.
- Added a distillation weights SG-TD3 critic-warm-3 manual-off runner.

## Notes

- The dedicated SG-TD3 weights runner disables behavioral cloning, handoff, release gate, TD3 authority ramp, reward probation, shadow identity diagnostics, and solve-failure identity fallback.
- The runner keeps only the nonfinite identity fallback as a numerical guard.
- The SG supervisor action is the identity weight multiplier, matching the shared `utils.weights_runner` SG-TD3 path.
