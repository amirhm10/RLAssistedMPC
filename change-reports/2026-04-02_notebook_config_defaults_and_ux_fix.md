# Notebook Config Defaults And UX Fix

## Summary
- Fixed the active unified notebooks so the user-editable run controls are explicit and no longer depend on hidden profile values.
- Added override variables for run size, plot windows, compare windows, naming, and baseline paths.
- Cleaned the opening/config flow so the parameter cells resolve defaults before they are used.

## Main Fixes
- Polymer unified RL notebooks now define fallback defaults for:
  - `n_tests`
  - `set_points_len`
  - `warm_start`
  - `test_cycle`
- The top config cells now expose optional overrides such as:
  - `RESULT_PREFIX_OVERRIDE`
  - `COMPARE_PREFIX_OVERRIDE`
  - `BASELINE_MPC_PATH_OVERRIDE`
  - `N_TESTS_OVERRIDE`
  - `SET_POINTS_LEN_OVERRIDE`
  - `WARM_START_OVERRIDE`
  - `TEST_CYCLE_OVERRIDE`
  - `PLOT_START_EPISODE_OVERRIDE`
  - `COMPARE_START_EPISODE_OVERRIDE`
- Distillation unified notebooks were aligned to the same pattern and cleaned so they use the resolved Aspen/data/results settings consistently.
- Plot/compare cells now use the resolved variables instead of bypassing overrides.

## Validation
- `nbformat` validation passed for all active unified runtime notebooks.
- Python AST parsing passed for all code cells in those notebooks.
- Short opening-cell smoke execution passed for:
  - `RL_assisted_MPC_horizons_unified.ipynb`
  - `RL_assisted_MPC_matrices_unified.ipynb`

## Scope
- This pass focused on the active unified runtime notebooks.
- Cache files, editor files, and archived notebooks were intentionally excluded.
