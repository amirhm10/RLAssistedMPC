# 2026-04-27 Behavioral Cloning Matrix Handoff

## Summary

Added a shared behavioral-cloning handoff surface for the scalar and structured
matrix supervisors, with actor-loss support in both TD3 and SAC.

## Main changes

- Added `utils/behavioral_cloning.py` to build the BC schedule, resolve the
  nominal-only target, and emit shared BC diagnostics.
- Extended the continuous training path so matrix-family runners can pass
  `bc_context` into `agent.train_step(...)`.
- Updated `TD3Agent/agent.py` and `SACAgent/sac_agent.py` to add the nominal
  BC penalty and return BC metadata.
- Wired BC into `utils/matrix_runner.py` and
  `utils/structured_matrix_runner.py` only.
- Enabled BC by default for polymer matrix and structured matrix, while
  disabling the prior matrix-step protections for the polymer isolation run.
- Added the same BC config surface to distillation matrix and structured matrix
  defaults, but kept it disabled by default.
- Updated the unified matrix notebooks to show BC in the run summary and pass
  the config through to the runners.
- Extended the matrix multiplier recovery report with the BC implementation
  progress and the chosen TD3/SAC loss equations.

## Validation

- Imported polymer and distillation defaults to confirm the intended enablement
  and disablement policy.
- Validated the four edited notebooks with `nbformat`.
- Compiled the modified Python modules with `py_compile`.
- Exercised TD3 and SAC `train_step(...)` with and without `bc_context`.
