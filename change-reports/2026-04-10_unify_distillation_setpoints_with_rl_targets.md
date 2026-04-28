## Summary

Removed the separate distillation baseline setpoint definition and aligned the
baseline distillation notebook with the shared RL supervisory setpoints.

## Changed Files

- `systems/distillation/config.py`
- `systems/distillation/notebook_params.py`
- `systems/distillation/__init__.py`
- `distillation_MPCOffsetFree_unified.ipynb`

## Details

- Deleted the standalone `DISTILLATION_BASELINE_SETPOINTS_PHYS` constant.
- Kept `DISTILLATION_RL_SETPOINTS_PHYS` as the single shared distillation
  supervisory setpoint pair.
- Updated the distillation notebook defaults so the system setup no longer
  exposes `baseline_setpoints_phys`.
- Updated `distillation_MPCOffsetFree_unified.ipynb` to use
  `SYS["rl_setpoints_phys"]`.

## Validation

- Python AST parse passed for the touched distillation modules.
- Notebook code compilation passed for
  `distillation_MPCOffsetFree_unified.ipynb`.
- Repo-wide search confirmed there are no remaining references to
  `baseline_setpoints_phys` or `DISTILLATION_BASELINE_SETPOINTS_PHYS`.
