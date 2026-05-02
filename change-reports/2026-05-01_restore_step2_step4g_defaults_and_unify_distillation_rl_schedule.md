# Restore Step 2 + Step 4G Defaults And Unify Distillation RL Schedule

Date: 2026-05-01

## Summary

Restored Step 2 + Step 4G as the live default matrix handoff path for polymer and distillation, disabled both Step 3 execution mechanisms in those shared defaults, and removed the last inline distillation horizon schedule duplication so the distillation RL notebooks resolve their episode schedule from one shared table.

## Changes

- Enabled behavioral cloning for polymer `matrix` and `structured_matrix`.
- Disabled `mpc_dual_cost_shadow` and `mpc_usefulness_gate` in the shared polymer matrix defaults.
- Preserved the previously working polymer Step 4G timing recipes:
  - scalar: freeze `2 / 2`, BC `0.3 / 20`, release guard `8 / 12`
  - structured: freeze `3 / 3`, BC `0.6 / 25`, existing structured label weights, release guard `10 / 15`
- Enabled behavioral cloning for distillation `matrix` and `structured_matrix`.
- Disabled `mpc_dual_cost_shadow` and `mpc_usefulness_gate` in the shared distillation matrix defaults.
- Standardized both distillation matrix families on the conservative Step 4G rollout:
  - freeze `5 / 5`
  - BC `lambda_bc_start = 0.3`
  - BC `active_subepisodes = 20`
  - no structured label-weight overrides
  - release guard `15 / 30`
- Replaced the inline `DISTILLATION_HORIZON_STANDARD_DEFAULTS["run_profiles"]` dictionary with `deepcopy(DISTILLATION_HORIZON_RUN_PROFILES)`.
- Added `report/scripts/verify_matrix_step4g_defaults.py` to assert the new matrix defaults and the shared distillation RL run-profile equality.

## Validation

- Ran `report/scripts/verify_matrix_step4g_defaults.py` in `rl-env`.
- Ran `py_compile` on `systems/polymer/notebook_params.py`, `systems/distillation/notebook_params.py`, and `report/scripts/verify_matrix_step4g_defaults.py`.
- Confirmed the distillation RL run-profile tables remain aligned at `n_tests = 200`, `set_points_len = 200`, `warm_start = 10`, `test_cycle = [False, False, False, False, False]`, `plot_start_episode = 2`, and `compare_start_episode = 2`.
