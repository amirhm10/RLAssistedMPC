# Distillation Reidentification Diagnostics Report

## What changed

- Added `report/scripts/generate_distillation_reidentification_report_assets.py`
  to extract metrics and generate figures for the April 20, 2026 distillation
  reidentification analysis.
- Added `report/distillation_reidentification_diagnostics_2026_04_20.tex`
  and compiled `report/distillation_reidentification_diagnostics_2026_04_20.pdf`.
- Added figure assets under
  `report/figures/distillation_reidentification_2026_04_20/`.

## Scope

The report compares the latest run `20260420_171837` against three earlier
distillation reidentification runs:

- `20260416_113030`
- `20260417_174602`
- `20260418_115020`

It focuses on setup changes, reward-gap behavior relative to MPC, blend
authority, accepted identification updates, and effective blended model
deviation.

## Main finding

The latest run shows that `id_window = 160` and `id_update_period = 20` do not
cause collapse by themselves. The critical factors were warm-start hidden drift
and excessive blend authority. Freezing identification during warm start and
holding a small fixed blend removes the collapse, but this is still an
isolation study rather than a learned-policy result because `force_eta_constant`
was enabled.
