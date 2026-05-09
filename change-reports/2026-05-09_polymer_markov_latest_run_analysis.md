# 2026-05-09 Polymer Markov Latest-Run Analysis

## What changed

- Added `report/scripts/analyze_polymer_markov_latest_run.py` to reproduce a latest-vs-previous Markov analysis from saved result bundles.
- Added a new report, `report/polymer_markov_latest_run_analysis_2026_05_09.md`.
- Generated a new dated analysis artifact folder, `report/figures/polymer_markov_latest_run_20260509/`, with:
  - reward-delta comparison figures
  - tail-output comparison figures
  - action-source and saturation figures
  - `z`-usage comparison figures
  - reward-rescoring comparison figure
  - summary CSV exports

## Main findings captured

- The latest unified run is almost nominal MPC under the shared unified reward.
- The previous prototype's late reward advantage does not survive rescoring with the shared unified reward.
- The strongest differences between the two implementations are the reward definition and the nominal comparison reference, not the `z` bounds.
