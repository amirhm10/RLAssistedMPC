# 2026-04-28 Distillation Observer Follow-Up Report Update

## Summary

Extended the distillation observer pole report with the latest focused follow-up sweep around `p19_uniform_fast`.

## What Changed

- Updated `report/distillation_observer_pole_sweep_2026_04_28.md` to cover:
  - the original broad sweep
  - the focused `p19` follow-up sweep from `20260428_141210`
- Added a dedicated follow-up figure pack under:
  - `report/figures/distillation_observer_followup_20260428/`
- Added a reproducible figure/script entrypoint:
  - `report/scripts/generate_distillation_observer_followup_report.py`

## Main Findings Captured

- `q01_uniform_fast_minus_04` is the best local follow-up candidate.
- `q11_front_faster_b` and `q10_front_faster_a` are the next strongest candidates.
- The local neighborhood around `p19` is useful, but the best follow-up candidates still do not beat `p00_old_aggressive_reference`.
- Slowing the last two poles remains strongly harmful.
