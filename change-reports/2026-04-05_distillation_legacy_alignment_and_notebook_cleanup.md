## Summary

This change refreshes the active distillation unified notebooks so their defaults, comments, and experiment setup are closer to the archived distillation notebooks while keeping the shared unified runners and plotting stack.

## What changed

- Added canonical legacy-aligned distillation constants in `systems/distillation/config.py`:
  - baseline setpoint pair
  - shared distillation observer poles
  - legacy-aligned run-profile maps for baseline, horizon, matrices, weights, residual, and combined
- Exported the new constants from `systems/distillation/__init__.py`.
- Rewrote the top setup/config cells of the distillation unified notebooks to be less crowded and more user-editable:
  - `distillation_systemIdentification_unified.ipynb`
  - `distillation_MPCOffsetFree_unified.ipynb`
  - `distillation_RL_assisted_MPC_horizons_unified.ipynb`
  - `distillation_RL_assisted_MPC_matrices_unified.ipynb`
  - `distillation_RL_assisted_MPC_weights_unified.ipynb`
  - `distillation_RL_assisted_MPC_residual_unified.ipynb`
  - `distillation_RL_assisted_MPC_combined_unified.ipynb`
- Fixed several migration regressions:
  - missing imports for the new legacy-aligned run-profile constants
  - missing imports for the shared distillation observer poles
  - duplicated `system_metadata` entries in runtime contexts
  - baseline `warm_start` was incorrectly hardcoded to `0` instead of using the legacy-aligned profile value
  - system-identification notebook had stale leftover path variables from an incomplete rewrite
- The matrix notebook now chooses the archived Aspen family by `AGENT_KIND` (`matrix_td3` vs `matrix_sac`) before resolving the default plant path.

## Legacy alignment now in place

- Baseline notebooks use the archived baseline setpoint pair:
  - `[0.013, -23.0]`
  - `[0.018, -22.0]`
- Horizon, matrices, weights, and residual use the archived RL setpoint pair:
  - `[0.013, -23.0]`
  - `[0.028, -21.0]`
- Combined uses the archived combined setpoint pair:
  - `[0.013, -23.0]`
  - `[0.018, -22.0]`
- Distillation observer poles now come from one shared canonical vector instead of repeated inline arrays.
- Legacy run sizes such as `n_tests`, `set_points_len`, `warm_start`, and combined `DECISION_INTERVAL=5` are centralized and reused.

## Validation

- All edited distillation unified notebooks pass `nbformat` validation.
- All code cells in the edited notebooks parse successfully with Python AST checks.
- The notebooks now reference the expected legacy-aligned distillation constants and no longer contain the stale rewrite artifacts that caused undefined-name issues.

## Notes

- This cleanup focuses on notebook structure/defaults and obvious legacy-alignment mismatches.
- The archived distillation subtree remains untouched.
- Distillation baseline and combined reward-family alignment can still be audited further if exact reward-range matching to the archived notebooks is required.
