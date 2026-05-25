# Distillation Residual Rho Authority Audit

Date: 2026-05-25

## Summary

Extended the controlled-authority failure report with a focused audit of residual rho authority across all canonical saved distillation residual unified runs. The audit was prompted by the temporary BC-handoff diagnostic default that disabled residual rho authority.

## Findings

- Rho has two separate roles in the residual runner: it can be appended to the mismatch-mode TD3 state, and it can also be used in the execution-time residual authority projection.
- The current post-BC-handoff defaults still append rho to the state, but disable rho-based execution authority.
- Historical residual results show that successful distillation residual TD3 runs usually relied on rho/headroom/deadband projection to shrink raw residual proposals into small executed corrections.
- Recommendation: keep rho authority active for the main distillation residual run, and reserve rho-off behavior for a named ablation.

## Files Added Or Updated

- `report/scripts/analyze_distillation_residual_rho_authority_20260525.py`
- `report/figures/distillation_residual_rho_authority_20260525/`
- `report/distillation_controlled_authority_failure_analysis_2026_05_24.md`

## Validation

- Analysis script rerun successfully with the project Conda environment.
- Figures visually inspected for readability.
- Compile and diff validation completed after the report update.
