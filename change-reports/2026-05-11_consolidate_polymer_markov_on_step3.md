# Consolidate Polymer Markov On Step 3

Date: 2026-05-11

## What changed

- consolidated polymer Markov down to one active notebook surface: `RL_assisted_MPC_markov_unified.ipynb`
- removed the retired legacy and legacy-mimic Markov notebook entrypoints
- removed the retired legacy and mimic Markov runtime files
- documented the shared polymer disturbance contract so Step 3 semantics are the canonical default across shared polymer live disturbed runners
- added a new starter report for the consolidated algorithm and future tuning

## Main result

- the Step 3 plant-step/disturbance semantics are now treated as the default shared polymer contract
- Markov no longer carries separate unified, legacy, and mimic execution surfaces
- the new baseline report for future tuning is `report/polymer_markov_unified_algorithm_and_tuning_start_2026_05_11.md`

## Files changed

- `utils/helpers.py`
- `RL_assisted_MPC_markov_unified.ipynb`
- `report/polymer_markov_unified_algorithm_and_tuning_start_2026_05_11.md`

## Files removed

- `polymer_markov_corrected_mpc_unified.ipynb`
- `polymer_markov_corrected_mpc_legacy.ipynb`
- `polymer_markov_corrected_mpc_unified_legacy_mimic.ipynb`
- `utils/markov_runner_legacy_mimic.py`
- `report/scripts/generate_polymer_markov_correction_assets_legacy.py`
