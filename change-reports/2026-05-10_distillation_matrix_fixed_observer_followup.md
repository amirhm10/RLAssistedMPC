# Distillation Matrix Fixed-Observer Follow-Up

Date: 2026-05-10

## Summary

- updated the distillation matrix-family follow-up script to analyze the true latest May 10 scalar and structured runs alongside the May 3 and May 8 phases
- generated a new dated figure bundle under `report/figures/distillation_matrix_structured_followup_20260510/`
- extended `report/distillation_matrix_structured_step4g_latest_2026_05_04.md` with the fixed-observer follow-up, cross-system comparison, and revised interpretation for future distillation Markov work

## Main Finding

The saved May 10 reruns are substantially worse than the May 8 observer-refresh runs, and the saved configuration diff isolates `recalculate_observer_on_matrix_change: True -> False` as the key recorded change. For the current distillation scalar and structured matrix families, fixed-observer operation is not a rescue path.

## Files Changed

- `report/scripts/generate_distillation_matrix_structured_followup_assets.py`
- `report/distillation_matrix_structured_step4g_latest_2026_05_04.md`
- `report/figures/distillation_matrix_structured_followup_20260510/`
