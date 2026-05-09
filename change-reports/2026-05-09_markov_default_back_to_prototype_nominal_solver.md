# 2026-05-09 Markov default back to prototype nominal solver

## Objective

Make the unified polymer Markov notebook default to the original prototype-style nominal online solve for the next test run.

## Files changed

- `systems/polymer/notebook_params.py`
- `utils/markov_runner.py`
- `polymer_markov_corrected_mpc_unified.ipynb`

## What changed

- Changed the Markov controller default `nominal_solver_mode` from `state_space_shared` to `lifted_g0_prototype`.
- Updated the shared Markov runner fallback so callers that omit the flag also default to `lifted_g0_prototype`.
- Updated the unified polymer Markov notebook fallback string to the same prototype default.

## Notes

- The Markov reward default remains `prototype_legacy`, so the notebook default is now prototype-style on both the reward and the nominal online solve.
- I did not run a full Markov training rollout in this change.
