# 2026-05-10 Markov Stage Logging And Legacy RL-Only Path

## What changed

- added shared Markov stage-diagnostics export helper in `utils/markov_diagnostics.py`
- extended `utils/markov_runner.py` to log nominal, requested, LS, and executed control-sequence traces plus first-move differences and nominal-cost margins
- extended `utils/plotting_core.py` so unified Markov notebook outputs now save the new diagnostics into both `input_data.pkl` and `markov_stage_diagnostics.csv`
- extended `report/scripts/generate_polymer_markov_correction_assets_legacy.py` with the same stage logs and CSV export
- changed the legacy Markov notebook default path to skip the pre-RL diagnostic phases and run the RL/live Markov phase directly via `run_pre_rl_diagnostics = False`

## Why

The saved bundles were sufficient for `z` norms, gain drift, and action-source fractions, but not for the full `z -> G -> U* -> u0 -> y` suppression analysis. In particular, first-move differences, full-sequence differences, and nominal-cost margins were computed transiently and then discarded.

## New saved diagnostics

Both unified and legacy Markov outputs now persist:

- nominal control sequence per step
- requested TD3 candidate control sequence per step
- LS candidate control sequence per step
- executed control sequence per step
- first-move differences versus nominal
- full-sequence difference norms versus nominal
- nominal cost, candidate-on-native cost, candidate-on-nominal cost
- nominal-cost margins and cost-guard pass flags

## Legacy behavior change

The legacy notebook still imports the same script entrypoint, but the default run mode no longer spends time on the separate phase-1/phase-2/phase-3 diagnostic rollouts before launching the RL/live controller. Those diagnostics can still be re-enabled with `run_pre_rl_diagnostics=True` in overrides if needed.

## Verification

- imported the modified modules successfully
- smoke-checked the new history shapes for shared and legacy paths
- ran the stage-diagnostics row builder on existing saved unified and legacy bundles to confirm the CSV export path handles real bundles cleanly
