# Distillation Transfer Plan Refresh

## Summary

Refreshed the distillation-column migration documents after rechecking the updated source notebooks in `DIstillation Column Case/RL_assisted_MPC_DL`.

## What Changed

- Updated the transfer analysis to reflect that:
  - `systemIdentification.ipynb` is now distillation-based
  - baseline MPC source notebooks are now:
    - `MPCOffsetFree.ipynb`
    - `MPCOffsetFreeDistRamp.ipynb`
    - `MPCOffsetFreeDistRampNoise.ipynb`
- Revised the migration scope to include residual support, even though there is no mature native residual distillation notebook family in the source project.
- Refined the naming-normalization plan around:
  - `system_dict_new` / `scaling_factor_new`
  - baseline MPC result file inconsistencies
- Updated the approval summary to reflect the new source-of-truth assessment and the recommended notebook set.

## Files

- `report/distillation_column_transfer_extensive.md`
- `report/distillation_column_transfer_approval.md`

## Notes

- This is still planning/documentation only.
- No distillation runtime code was migrated in this step.
- Unrelated local data, result folders, and cache files were intentionally left out of the staged change.
