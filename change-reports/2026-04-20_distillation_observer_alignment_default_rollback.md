# Distillation Observer Alignment Default Rollback

Date: 2026-04-20

## Summary

Rolled the distillation notebook defaults for `observer_update_alignment` back to the legacy delayed-observer update path.

## Reason

The current-measurement corrector default caused instability in the distillation residual workflow:

- observer state overflow
- invalid values propagating through `xhatdhat`
- downstream `NaN` warnings in the MPC prediction path

The distillation mismatch feature construction already uses the current measured output in the RL state path, so reverting the observer-state update default does not remove the newer innovation/tracking features the user wanted to keep.

## File

- [systems/distillation/notebook_params.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/distillation/notebook_params.py)
