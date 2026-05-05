# Polymer Markov Correction Prototype

## Summary

Added a polymer-only prototype for prediction-error-validated Markov-parameter correction of offset-free MPC.

## Changes

- Added `polymer_markov_corrected_mpc_unified.ipynb` as the sectioned notebook entrypoint.
- Added `report/scripts/generate_polymer_markov_correction_assets.py` with local prototype helpers, closed-loop smoke execution, plotting, CSV summaries, report writing, and result-bundle export.
- Added `report/polymer_markov_correction_progress.md` as the living progress report.
- Generated smoke-test figures and summaries under `report/figures/polymer_markov_correction_20260505/`.

## Verification

Ran a capped smoke execution with `--max-steps 30` using the current disturbed polymer defaults otherwise unchanged. The run completed and saved a verification table. The full default remains `n_tests=200`, `set_points_len=400`, `predict_h=9`, and `cont_h=3`.

## Notes

The smoke run does not prove the method works. It only verifies that the implementation path, lifted equivalence check, Markov scoring, LS correction, live correction gate, plotting, report writing, and bundle saving execute on a short horizon.
