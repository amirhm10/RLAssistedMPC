# Distillation Matrix Wide Search With Step 4G And Step 3C Shadow

Date: 2026-04-29

## Summary

Restored the distillation scalar matrix notebook to the wide `A,B` multiplier search and switched the default handoff stack to:

- Step 1 offline multiplier diagnostics: enabled
- Step 2 release-protected advisory caps: enabled
- Step 3C dual-cost shadow diagnostics: enabled for logging only
- Step 3D usefulness gate: disabled
- Step 4G behavioral-cloning anchor plus release guard: enabled

## Files Changed

- `systems/distillation/notebook_params.py`
- `distillation_RL_assisted_MPC_matrices_unified.ipynb`

## Notes

- The notebook-local temporary override that pinned both `B` columns to nominal was removed.
- The notebook run summary now prints the active low/high matrix bounds so the restored wide search is visible before execution.
- Validation completed with `py_compile` for `systems/distillation/notebook_params.py` and `nbformat` validation for `distillation_RL_assisted_MPC_matrices_unified.ipynb`.
