# Distillation Structured Matrix Step 2 Wide Defaults

## Summary

The distillation structured-matrix notebook is already wired like the plain matrix notebook: Step 1 offline sensitivity diagnostics provide advisory bounds, Step 2 release-protected advisory caps are passed into the runner, and Step 3 MPC acceptance/fallback remains disabled by default. This update restores the structured distillation A-side default to the wide capped search range as requested.

## Changes

- Restored structured distillation `A` override bounds to wide defaults:
  - `a_low_override = 0.75`
  - `a_high_override = min(1.25, distillation_alpha_cap) = 1.1929`
- Kept structured `B` override bounds wide at `0.75` to `1.25`.
- Kept Step 2 release-protected advisory caps enabled.
- Kept Step 3 MPC acceptance/fallback disabled.

## Validation

- Import distillation defaults and assert the plain and structured matrix methods both use wide raw bounds, Step 2 enabled, and Step 3 disabled.
- Validate both distillation matrix notebooks with `nbformat`.
