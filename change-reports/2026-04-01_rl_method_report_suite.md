# RL Method Report Suite

## Date

2026-04-01

## Summary

- Added a new root `report/` directory for method-level documentation of the
  RL-assisted notebook families.
- Created five standalone LaTeX reports and five matching Markdown summaries.
- Grouped the current method families into:
  - horizon adaptation
  - weight adaptation
  - model multipliers plus observer poles
  - residual methods
  - combined multi-agent supervision
- Added shared TeX includes for common formatting and notation.
- Planned output includes compiled PDFs for each report while excluding transient
  LaTeX build files from the final tree.

## Files Intended For This Commit

- `report/_common_preamble.tex`
- `report/_common_notation.tex`
- `report/01_horizon_adaptation.tex`
- `report/01_horizon_adaptation.md`
- `report/02_weight_adaptation.tex`
- `report/02_weight_adaptation.md`
- `report/03_model_multipliers_and_poles.tex`
- `report/03_model_multipliers_and_poles.md`
- `report/04_residual_methods.tex`
- `report/04_residual_methods.md`
- `report/05_combined_supervisor.tex`
- `report/05_combined_supervisor.md`
- compiled PDFs for the five reports
- `change-reports/2026-04-01_rl_method_report_suite.md`
