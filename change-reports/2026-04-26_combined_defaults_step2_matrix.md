# Combined Defaults And Step 2 Matrix Guard

Date: 2026-04-26

## Summary

Rewrote the polymer and distillation combined workflow defaults so the combined notebooks inherit the current single-agent horizon, scalar matrix, weights, and residual settings instead of carrying a separate drifted configuration.

## Changes

- Disabled matrix Step 3 `mpc_acceptance_fallback` by default for scalar and structured matrix methods.
- Kept matrix methods on Step 2 release-protected advisory caps, with Step 1 scalar sensitivity diagnostics feeding the advisory bounds.
- Added Step 1/Step 2 scalar matrix diagnostic setup to both combined notebooks.
- Updated the combined runner to clip matrix multipliers through the Step 2 release schedule before building the assisted MPC prediction model.
- Added combined result logs for matrix policy, candidate, executed multipliers, release bounds, phase, clip fraction, and ramp fraction.
- Updated combined defaults to use per-agent TD3/SAC config dictionaries for matrix, weights, and residual agents.
- Aligned polymer combined episode length with single-agent defaults and aligned distillation combined decision interval with the horizon default.

## Validation

- Notebook JSON validation and import/config assertions should be run with the `rl-env` Python interpreter.
- Polymer full-run execution is the default runtime validation path; distillation validation should stay at smoke/config level unless a full Aspen run is explicitly requested.
