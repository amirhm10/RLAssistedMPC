# Analyze Distillation Residual Tail Degradation

Date: 2026-04-25

## Summary

Extended the ongoing matrix-multiplier recovery report with a focused diagnosis of why the latest distillation residual TD3 run degrades in the final episodes.

## Findings

- The residual run briefly beats MPC in episodes 61-100, then degrades in episodes 101-200.
- Tail degradation is mainly associated with worse x24 composition tracking, a lower reward bonus, and lower inside-band weight.
- The raw residual policy becomes more aggressive in the tail, especially on input 2, but the `rho` authority projection clips most of the requested correction.
- Authority projection fraction has the strongest negative episode-level correlation with reward delta among the inspected diagnostics.
- The residual safety layer is doing its safety job, but persistent projection means the actor is learning raw actions outside the useful executable envelope.

## Files

- `report/matrix_multiplier_cap_calculation_and_distillation_recovery.md`
- `report/figures/distillation_residual_tail_20260425/distillation_residual_tail_episode_diagnostics.png`
- `report/figures/distillation_residual_tail_20260425/distillation_residual_tail_window_summary.png`
- `report/figures/distillation_residual_tail_20260425/distillation_residual_tail_last_episode.png`
- `report/figures/distillation_residual_tail_20260425/distillation_residual_tail_correlations.png`
