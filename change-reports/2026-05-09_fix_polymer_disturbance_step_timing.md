## Summary

Updated `utils/helpers.py` so dictionary-style disturbance schedules are applied to the polymer plant before `system.step()` instead of after it.

## Why

The old polymer Markov prototype and older polymer MPC code apply `Qi`, `Qs`, and `hA` before stepping the nonlinear plant. The shared helper had drifted to a one-step-late disturbance application for dict-style schedules, which changed the closed-loop semantics for shared polymer runners.

## Scope

This change affects shared runners that call `step_system_with_disturbance(...)`, including the Markov, MPC baseline, matrix, structured-matrix, weights, residual, horizon, combined, and reidentification runners.

## Verification

- Parsed the edited helper successfully.
- Ran a dummy-system validation to confirm dict-style disturbances are now visible inside `system.step()`.
