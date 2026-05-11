# 2026-05-11 Restore Markov Step 3 Default Release

## What changed

- removed the hard notebook override `WARM_START_OVERRIDE = 10` from `RL_assisted_MPC_markov_unified.ipynb`
- restored the notebook default to the actual Step 3 consolidated surface, which uses the Markov family default `warm_start = 0`
- updated `report/polymer_markov_unified_algorithm_and_tuning_start_2026_05_11.md` so it distinguishes the Step 3 default from the later LS-warm-start tuning branch

## Why

The deleted Step 3 legacy-mimic notebook did not force a 10-episode LS warm start. Adding that override in the consolidated notebook changed the runtime surface, so the new notebook was no longer reproducing the Step 3 run that had matched legacy trajectory behavior.
