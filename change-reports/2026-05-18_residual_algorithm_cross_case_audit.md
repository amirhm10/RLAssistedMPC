# Residual Algorithm Cross-Case Audit

Date: 2026-05-18

## Summary

Created a new report auditing the residual algorithm across polymer and distillation:

`report/residual_algorithm_polymer_distillation_2026_05_18.md`

## Findings

- Polymer and distillation residual notebooks use the same shared runner:
  `utils/residual_runner.py`.
- Both pass rho authority settings into `residual_cfg`.
- Saved bundles for both case studies contain rho/projection diagnostics.
- The live distillation log shows a release shock at subepisode `16`, exactly after warm start plus five frozen residual-action subepisodes.
- Rho is functional but magnitude-only; it does not check whether a residual direction improves reward or tracking.

## Added

- `report/scripts/analyze_residual_algorithm_polymer_distillation_20260518.py`
- figures under `report/figures/residual_algorithm_polymer_distillation_20260518/`
- `report/residual_algorithm_polymer_distillation_2026_05_18.md`

## Validation

- Loaded saved polymer and distillation residual bundles.
- Parsed the pasted live distillation residual log.
- Generated figures and summary JSON.
- No notebook execution was performed.
- No Aspen execution was performed.
