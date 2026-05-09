# 2026-05-09 Markov nominal solver mode A/B switch

## Objective

Add a single runtime flag so the unified polymer Markov workflow can switch the nominal online MPC reference solve between:

- `state_space_shared`: the current unified `MpcSolverGeneral.mpc_opt_fun(...)` path
- `lifted_g0_prototype`: the pre-migration prototype path that solves the nominal action with `solve_lifted_mpc(..., G0, ...)`

## Files changed

- `systems/polymer/notebook_params.py`
- `utils/markov_runner.py`
- `polymer_markov_corrected_mpc_unified.ipynb`

## What changed

- Added `controller["nominal_solver_mode"] = "state_space_shared"` to `POLYMER_MARKOV_DEFAULTS`.
- Added runner helper `solve_nominal_reference_step(...)` in `utils/markov_runner.py`.
- Switched the live Markov closed loop to choose the nominal solve from `nominal_solver_mode` each step.
- Threaded the flag through the unified polymer Markov notebook summary and `markov_cfg`.
- Added the selected nominal solver mode to the saved Markov summary/result bundle.

## Validation

- Parsed `utils/markov_runner.py` and `systems/polymer/notebook_params.py` successfully with Python `ast`.
- Loaded `polymer_markov_corrected_mpc_unified.ipynb` as JSON successfully after the notebook edit.
- Exercised both nominal solver modes on the same synthetic MPC problem and confirmed both branches return successful solutions.

## Notes

- I did not run a full polymer training rollout yet.
- The sandbox could not access the configured `rl-env` interpreter directly, so the smoke check used the local Miniconda base Python for parsing and branch validation only.
