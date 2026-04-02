# Fix Combined Compare Output Directory

## Summary

Fixed `plot_combined_results(...)` so its built-in baseline comparison no longer tries to create the compare output inside the timestamped RL run directory.

## Problem

The combined plotting path was doing this:

- create RL run output directory
- save the RL bundle there
- call the baseline compare helper with `directory=out_dir`

That caused the compare helper to create another timestamped directory inside the RL run directory. With the long combined prefixes used for enabled-agent/state-mode tags, Windows path length limits could be exceeded and `os.makedirs(...)` failed with `FileNotFoundError`.

## Fix

- `utils/plotting_core.py`
  - `plot_combined_results_core(...)` now uses:
    - `plot_cfg["compare_directory"]` when provided
    - otherwise the top-level plot `directory`
  - it no longer defaults the compare output directory to the timestamped RL run directory

- `RL_assisted_MPC_combined_unified.ipynb`
  - now passes `compare_directory = REPO_ROOT / "Result"`

- `distillation_RL_assisted_MPC_combined_unified.ipynb`
  - now passes `compare_directory = RESULT_DIR`

## Validation

Validated in `rl-env` by:

- compiling the patched plotting core
- notebook schema and per-cell syntax validation for both combined unified notebooks
- running a real combined-plot smoke test against an existing saved bundle with baseline comparison enabled

The smoke run created:

- one RL output directory
- one sibling compare directory

instead of nesting the compare directory under the RL run directory.
