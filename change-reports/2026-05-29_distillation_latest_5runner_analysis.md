# 2026-05-29 Distillation Latest Five-Runner Analysis

Created a focused research analysis report for the latest distillation disturbance-fluctuation runs across all five active runners: weights, horizon, dueling horizon, residual, and Markov.

Artifacts added:

- `report/distillation_latest_5runner_safety_analysis_2026_05_29.md`
- `report/scripts/analyze_distillation_latest_settings_20260529.py`
- `report/figures/distillation_latest_settings_20260529/`

Main findings:

- Residual no-rho is the strongest late performer, but has severe early release shock.
- Weights improved strongly under BC handoff and should keep live authority with candidate screening.
- Horizon and dueling horizon remain active, safer discrete baselines.
- Markov failed because priority fallback, reward probation, LS fallback, and candidate vetoes were disabled while TD3 execution was forced.

Validation:

- Ran the analysis script successfully and regenerated metrics plus figures from saved bundles only.
- Visually inspected the main reward, batch-comparison, tail-tracking, and Markov-safety figures.
