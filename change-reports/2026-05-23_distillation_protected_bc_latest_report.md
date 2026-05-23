# 2026-05-23 Distillation Protected-BC Latest Runs Report

## Scope

Analyzed the latest completed active distillation family runs from 2026-05-22: weights, residual, Markov, horizon DDQN, and dueling horizon. Combined was not included because it was not part of this completed batch.

## Outputs

- Added `report/distillation_protected_bc_latest_family_runs_2026_05_23.md`.
- Added `report/scripts/analyze_distillation_protected_bc_latest_20260523.py`.
- Generated report figures and metric tables under `report/figures/distillation_protected_bc_latest_20260523/`.

## Main Finding

The continuous TD3 families were safe but did not meaningfully receive live authority. Weights, residual, and Markov all had protected-BC release step `-1` and post-warm released fraction `0.0`. Dueling horizon was the strongest active learned controller in the batch.

## Validation

- Recomputed latest-run metrics from saved result bundles using current distillation reward defaults.
- Audited the generated figures visually for readability.
- No full Aspen simulation was run.
