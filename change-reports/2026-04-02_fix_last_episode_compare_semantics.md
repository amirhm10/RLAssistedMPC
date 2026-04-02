# Fix Last-Episode Comparison Semantics

## Summary

- Fixed the shared MPC-vs-RL comparison helper so the "last episode" figures now compare the actual final episode of the RL run against the actual final episode of the MPC baseline.

## Problem

- The shared helper in `utils/plotting_core.py` was producing a tail-of-window comparison for the second compare figure.
- That behavior drifted from the older notebook-era helpers, which explicitly compared:
  - final RL episode
  - against final MPC episode
- The mismatch was especially confusing in nominal mode because the figure name implied episode-level comparison while the implementation was using the tail of the currently selected overlap window.

## Fix

- `compare_mpc_rl_from_dirs_core(...)` now computes:
  - `last_episode_steps = min(time_in_sub_episodes, steps_rl, steps_mpc, nFE_sp)`
- It then slices:
  - the final `last_episode_steps + 1` output samples from RL and MPC
  - the final `last_episode_steps` input samples from RL and MPC
  - the final `last_episode_steps` setpoint samples
- The saved filenames now match the restored semantics:
  - `compare_outputs_last_episode`
  - `compare_inputs_last_episode`

## Validation

- Python compile check for `utils/plotting_core.py`
- Real nominal horizon comparison smoke test against:
  - `Data/horizon_nominal_unified/.../input_data.pkl`
  - `Data/mpc_results_nominal.pickle`
- Verified that the last-episode compare call completes successfully with the restored slicing logic
