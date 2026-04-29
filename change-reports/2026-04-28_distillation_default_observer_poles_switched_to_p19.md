# 2026-04-28 Distillation Default Observer Poles Switched To `p19`

## Summary

Changed the shared distillation observer-pole default to:

`[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]`

This now flows through the distillation baseline and RL notebook families via the shared distillation config.

## Scope

- Updated `systems/distillation/config.py`
- Removed the notebook-local structured-matrix pole override from `distillation_RL_assisted_MPC_structured_matrices_unified.ipynb`
- Deleted the temporary sweep notebook `distillation_MPCOffsetFree_observer_pole_sweep_temp.ipynb`

## Notes

- This is a default-policy change. It does not rewrite historical result bundles or reports.
- Existing distillation notebooks that read `SYS["observer_poles"]` will now use the new default automatically.
