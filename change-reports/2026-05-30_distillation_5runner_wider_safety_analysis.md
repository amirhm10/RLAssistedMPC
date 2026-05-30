# Distillation Five-Runner Wider Safety Analysis

Date: 2026-05-30

## Summary

Added a reproducible analysis pass for the May 29 distillation five-runner simulations after the wider-search and safety-restoration changes.

## Files Added

- `report/scripts/analyze_distillation_wider_safety_20260530.py`
- `report/distillation_5runner_wider_safety_analysis_2026_05_30.md`
- `report/figures/distillation_wider_safety_20260530/analysis_summary.json`
- `report/figures/distillation_wider_safety_20260530/current_summary_metrics.csv`
- `report/figures/distillation_wider_safety_20260530/current_vs_baseline_metrics.csv`
- `report/figures/distillation_wider_safety_20260530/current_vs_previous_metrics.csv`
- `report/figures/distillation_wider_safety_20260530/current_horizon_pair_counts.csv`
- `report/figures/distillation_wider_safety_20260530/*.png`

## Main Findings

- Residual is still the only latest runner clearly above OF-MPC, with tail-20 reward 24.43 versus OF-MPC 13.90.
- Residual early safety improved strongly versus the May 28 run, but the saved bundle shows `residual_safety_enabled=False`, so the intended reward-probation and shadow-rho safety config was not actually passed through.
- Markov safety restoration improved tail reward by 18.91 versus the May 28 forced-TD3 run, but Markov remains below baseline because candidate-quality diagnostics are still poor.
- Weights, standard horizon, and dueling horizon are safer or more diagnostic, but tail reward dropped after the wider-search and cooldown/probation changes.

## Validation

- Analysis script was run against saved bundles only.
- Generated figures were checked for nonblank image statistics.
- No full Aspen simulation was run.
