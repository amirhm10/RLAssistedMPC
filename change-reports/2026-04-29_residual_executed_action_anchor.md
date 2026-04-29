# Residual Executed-Action Anchor

## Summary

Added executed-action behavioral cloning for the shared residual supervisor used by the polymer and distillation residual notebooks. The actor can now be anchored to the residual action that actually reached the plant after projection, using the existing TD3/SAC BC loss path instead of a residual-specific loss implementation.

## Files Changed

- `utils/behavioral_cloning.py`
- `utils/residual_runner.py`
- `systems/polymer/notebook_params.py`
- `systems/distillation/notebook_params.py`
- `RL_assisted_MPC_residual_unified.ipynb`
- `distillation_RL_assisted_MPC_residual_unified.ipynb`

## Main Changes

- Added `behavioral_cloning.target_mode = "executed_action"` support and `action_gap_tolerance`.
- Added a BC schedule start-step override so residual TD3 can start anchoring at the first live post-freeze action.
- Reused the shared BC logging surface and added generic target-distance logging.
- Threaded residual BC through `utils/residual_runner.py` with deterministic `act_eval(...)` policy logging, executed-action targets, and `policy_executed_gap_norm_log`.
- Enabled residual BC defaults for both polymer and distillation notebook parameter tables.
- Wired both residual notebooks to read, summarize, and pass the residual BC config into `residual_cfg`.

## Validation

- `py_compile` passed for:
  - `utils/behavioral_cloning.py`
  - `utils/residual_runner.py`
  - `systems/polymer/notebook_params.py`
  - `systems/distillation/notebook_params.py`
- Both residual notebooks were validated with `nbformat`.
- Polymer residual smoke validation passed with shortened TD3 settings:
  - BC enabled bundle reported `target_mode=executed_action`
  - BC schedule `start_step` matched `phase1_first_live_action_step`
  - `bc_active_log` contained active steps
  - `policy_action_raw_log`, `executed_action_raw_log`, `bc_policy_target_distance_log`, and `policy_executed_gap_norm_log` were present
  - steps within the configured action-gap tolerance kept `bc_active_log == 0`
- Distillation validation was limited to default-surface and notebook-wiring checks; Aspen was not launched.
