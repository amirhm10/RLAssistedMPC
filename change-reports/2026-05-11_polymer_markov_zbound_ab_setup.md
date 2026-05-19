# 2026-05-11 Polymer Markov z-bound A/B setup

## Goal

Set up the next polymer Markov experiment to test whether the remaining late TD3 decline is mainly caused by action-range saturation.

## Notebook change

Updated `RL_assisted_MPC_markov_unified.ipynb` so a one-off notebook override can vary only the Markov correction range:

- `Z_BOUND_OVERRIDE = None` by default
- set `Z_BOUND_OVERRIDE = 0.05` for the lower-range run
- set `Z_BOUND_OVERRIDE = 0.08` for the higher-range run

When the override is set, the notebook now auto-tags the result prefixes:

- `..._zbound_005`
- `..._zbound_008`

This is designed to compose cleanly with the replay override suffixes if needed.

## Bundle metadata

Updated `utils/markov_runner.py` so saved bundles record:

- `markov_z_bound`

This makes the A/B provenance explicit in the saved `input_data.pkl`.

## Analysis support

Added `report/scripts/analyze_polymer_markov_zbound_ab.py` to compare two saved runs on:

- TD3 accepted fraction by episode
- requested TD3 score by episode
- LS score by episode
- raw-action saturation by episode
- `max |z_TD3|`
- `max |z_LS|`
- `||z_TD3 - z_LS||`
- average reward

## Expected interpretation

- If TD3 share rises with the larger `z_bound`, the remaining bottleneck is mainly authority saturation.
- If TD3 share still collapses, the next suspect is the actor objective rather than replay or state conditioning.
