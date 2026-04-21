# Legacy Observer Default With Current-Measurement Mismatch Features

Date: 2026-04-20

## Summary

Set the default observer update alignment back to the legacy path for both polymer and distillation, while keeping the mismatch-feature construction on the current measured output.

## Reason

The desired default is:

- observer state propagation uses the legacy delayed update logic
- innovation and tracking features still reflect the current measurement seen by the RL policy

That matches the existing shared state-feature flow more closely and avoids forcing the newer predictor-corrector observer update as the notebook default.

## Files

- [systems/polymer/notebook_params.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/polymer/notebook_params.py)
- [systems/distillation/notebook_params.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/distillation/notebook_params.py)
- [report/method_scoped_conditioning_change_log.md](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/report/method_scoped_conditioning_change_log.md)
