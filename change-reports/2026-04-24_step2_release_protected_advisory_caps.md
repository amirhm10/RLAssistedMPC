# Step 2 Release-Protected Advisory Caps

Date: 2026-04-24

## Summary

Implemented unified Step 2 release-protected advisory caps for matrix and structured-matrix supervisors.

## Changes

- Added `utils/multiplier_release_schedule.py` for phase-dependent release bounds, clipping, log-space authority ramps, and inverse raw-action mapping.
- Wired scalar matrix and structured matrix runners to:
  - keep the actor action space wide,
  - map the requested policy action to wide multipliers,
  - clip executed multipliers through the Step 2 schedule,
  - use executed multipliers in MPC and plant rollout,
  - store executed raw actions in replay by default,
  - log requested and executed multiplier behavior separately.
- Added `release_protected_advisory_caps` defaults:
  - polymer matrix and structured matrix enabled,
  - distillation matrix and structured matrix disabled.
- Wired all four matrix notebooks to extract Step 1 diagnostic `suggested_bounds` when Step 2 is enabled.
- Updated the ongoing matrix multiplier report from Option naming to Step naming and added Step 2 progress placeholders.

## Validation

- Imported the new release schedule utility.
- Ran unit-style schedule, clipping, and inverse-mapping checks.
- Validated all four edited notebook JSON files.
- Confirmed polymer Step 2 defaults are enabled and distillation defaults are disabled.
- Compiled the new release schedule module plus the edited runners and default modules with `py_compile`.
