# 2026-05-09 Markov restore shared nominal default and seed TD3

## Objective

Make the next polymer Markov rerun a cleaner A/B by:

- restoring the default nominal online solve to the shared `state_space_shared` path
- keeping the temporary `prototype_legacy` reward in place
- adding deterministic TD3 seeding and saving that seed in the run bundle

## Files changed

- `systems/polymer/notebook_params.py`
- `TD3Agent/agent.py`
- `utils/markov_runner.py`
- `polymer_markov_corrected_mpc_unified.ipynb`

## What changed

- Restored `controller["nominal_solver_mode"]` to `state_space_shared` for the Markov defaults.
- Added `seed: 7` to the Markov TD3 agent defaults.
- Wired the TD3 seed through `make_td3_markov_agent(...)` into `TD3Agent`.
- Added deterministic seeding for `random`, `numpy`, and `torch` inside `TD3Agent`.
- Recorded `td3_seed` in the Markov summary/result bundle.
- Exposed the TD3 seed in the unified Markov notebook summary block.

## Notes

- The Markov reward default remains `prototype_legacy`.
- I did not run a full Markov training rollout in this change.
