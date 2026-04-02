# Fix Nominal MPC Comparison Window

## Summary

- Fixed the shared comparison helper used by the unified horizon, matrix, weight, and residual notebooks.
- The failure happened when a long RL run was compared against `Data/mpc_results_nominal.pickle`, which only stores two episodes.

## Root Cause

- `compare_mpc_rl_from_dirs_core(...)` normalized the nominal MPC pickle against the RL bundle and then used the RL-sized `nFE` to build the comparison window.
- That made the compare code build a long time axis while slicing only the shorter nominal MPC trajectory, which broke the last notebook cell during plotting.

## Fix

- The shared compare core now computes the comparison window from the actual stored trajectory lengths:
  - RL output length
  - MPC output length
  - setpoint length
- The start step is clamped to the valid overlap window before plotting.
- Nominal reward comparison behavior is unchanged: it still uses the last available nominal MPC episode reward as a constant reference line.

## Validation

- Reproduced the failure scenario against an existing unified nominal run with `start_episode=2`.
- Re-ran the shared comparison after the fix in `rl-env`.
- Confirmed the nominal compare path now completes successfully without changing the unified notebooks themselves.
