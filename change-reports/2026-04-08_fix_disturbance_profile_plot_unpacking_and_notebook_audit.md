# Fix Disturbance Profile Plot Unpacking And Notebook Audit

## Summary
- Fixed a shared plotting bug that broke disturbance-profile figures in some unified supervisor result plots.
- Re-audited the active unified notebooks for stale baseline MPC path references.
- Confirmed the saved notebook sources no longer use `RUN_PROFILE["baseline_mpc_path"]`.

## Root Cause
`disturbance_plot_items(...)` returns triples in the form:
- `(key, label, series)`

Several shared plotting paths were still unpacking them as:
- `(label, series)`

That mismatch caused errors like:
- `ValueError: too many values to unpack (expected 2)`

## Code Change
- Updated [utils/plotting_core.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/plotting_core.py)

Fixed disturbance-profile plotting loops in:
- matrix multiplier plots
- weight multiplier plots
- residual supervisor plots
- combined supervisor plots

They now consistently unpack:
- `(_, label, series)`

## Notebook Audit
I checked the active unified notebooks and confirmed there are no saved source references to:
- `RUN_PROFILE["baseline_mpc_path"]`
- `RUN_PROFILE['baseline_mpc_path']`

The active saved notebooks are already using the resolved:
- `BASELINE_MPC_PATH`

So the residual KeyError you saw is not present in the current notebook file on disk. That error was consistent with running an older in-memory notebook state before the later baseline-path cleanup.

## Validation
- `py_compile` passed for [utils/plotting_core.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/plotting_core.py)
- `nbformat` + `ast` parsing passed for the active unified notebooks
- real plotting smoke check using saved result bundles passed for:
  - matrix
  - weights
  - residual
  - combined

## Notes
- The smoke test created an untracked local folder under [Polymer/Results/plotting_disturbance_smoke](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/Polymer/Results/plotting_disturbance_smoke).
- Unrelated local data/result/cache files in the worktree were left untouched.
