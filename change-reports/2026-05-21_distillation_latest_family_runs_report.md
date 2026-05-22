# Distillation Latest Family Runs Report

Date: 2026-05-21

## Summary

Generated an extensive latest-run analysis report for the disturbed distillation column family runs: OF-MPC, TD3 weights, TD3 residual, horizon DDQN, dueling horizon DDQN, and TD3 Markov. The combined supervisor is excluded because it was not run in this batch.

## Outputs

- Report: `report/distillation_latest_family_runs_2026_05_21.md`
- Analysis script: `report/scripts/analyze_distillation_latest_family_runs_20260521.py`
- Figures and metric tables: `report/figures/distillation_latest_family_runs_20260521/`

## Main Finding

TD3 Weights is the strongest latest run by current-reward tail reward and by both output RMSE metrics. TD3 Markov remains competitive under `z_bound = 0.04` and active z-safety, but it has the largest T85 band-normalized tail error in this batch. TD3 Residual shows a late reward collapse with persistent residual projection activity.

## Validation

The report generator was rerun after the warm-start bookkeeping fix, and the generated figures were visually checked for readability and correct setpoint scaling.
