# Combined plotting parity and explicit MPC comparison

## Summary

Updated the combined RL-assisted MPC plotting path so combined runs produce the same style of per-agent diagnostic figures expected from the single-agent notebooks. The combined notebooks now call the MPC comparison explicitly after saving the RL plots, matching the single-agent notebook workflow.

## Changes

- Added combined parity plots for horizon, matrix, weight, and residual agents in `utils/plotting_core.py`.
- Added matrix Step 2 release diagnostics for phase, guard activity, clipping, ramping, and effective bounds when those logs are present.
- Made combined per-agent mismatch and training diagnostics available whenever the result bundle contains the data, instead of only in debug style.
- Updated polymer and distillation combined notebooks to call `compare_mpc_rl_from_dirs` directly and print both RL and comparison output directories.

## Validation

- Notebook JSON validation and Python compile checks were run after the edits.
- A lightweight synthetic combined plotting smoke check was used to verify the new figure names without running polymer training or opening Aspen.
