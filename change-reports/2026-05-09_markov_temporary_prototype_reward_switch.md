# 2026-05-09 Markov Temporary Prototype Reward Switch

## What changed

- Added `make_reward_fn_prototype_legacy(...)` to `utils/rewards.py`.
- Updated `polymer_markov_corrected_mpc_unified.ipynb` to select the Markov reward from `NB["reward"]["mode"]`.
- Set `POLYMER_MARKOV_DEFAULTS["reward"]["mode"] = "prototype_legacy"` in `systems/polymer/notebook_params.py` for a temporary A/B test.

## Why

This temporary switch lets the unified Markov notebook use the same reward shape as the pre-migration prototype so we can test whether reward design is the main cause of the behavioral gap.

## Revert path

To return to the shared unified reward, set:

- `POLYMER_MARKOV_DEFAULTS["reward"]["mode"] = "shared_relative_qr"`

and leave the helper switch in place for future A/B checks.
