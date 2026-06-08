# Polymer Baseline Final Comparison Alignment

## Summary

Aligned the exported OF-MPC baseline runner with the active polymer RL runner scenario before the final baseline rerun.

## Changes

- `MPCOffsetFree_unified.py` now uses `RUN_MODE = NB["run_mode"]`, so the shared default `disturb` mode is honored.
- The baseline runner now uses `SYS["rl_setpoints_phys"]`, matching horizons, weights, residual, Markov, and combined polymer runners.
- The disturbed baseline profile now uses `warm_start = 10`, matching the active RL episode schedule metadata.
- Added a baseline default guard test so the disturbed OF-MPC profile stays aligned with the active RL episode schedule.

## Validation

No full MPC rollout was run. Validation was limited to static/default checks and Python compilation.
