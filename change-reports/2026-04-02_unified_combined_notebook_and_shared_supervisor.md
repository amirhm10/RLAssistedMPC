# Unified Combined Notebook And Shared Supervisor

## Summary

- Added `RL_assisted_MPC_combined_unified.ipynb` as the canonical combined entrypoint.
- Added `utils/combined_runner.py` to orchestrate horizon, matrix, weight, and residual agents in one shared rollout.
- Added `plot_combined_results(...)` and the combined plotting core, including hybrid/paper/debug styling and combined-agent diagnostics.

## Main Changes

- Combined runtime now supports:
  - shared `RUN_MODE` for nominal vs disturbance
  - independent agent enable/disable toggles
  - per-agent state-mode selection
  - TD3/SAC selection for matrix, weight, and residual agents
  - optional `rho` authority in residual mismatch mode
- The combined result bundle uses namespaced logs for:
  - horizon traces and action indices
  - matrix multipliers
  - weight multipliers
  - residual raw/executed corrections
  - per-agent mismatch diagnostics
  - per-agent training diagnostics
- Combined plotting now covers:
  - outputs, inputs, rewards, tracking error, and observer overlays
  - per-agent decision traces
  - horizon/matrix/weight/residual diagnostics
  - baseline MPC comparison from the saved combined result directory

## Validation

- Python compile checks for:
  - `utils/combined_runner.py`
  - `utils/plotting_core.py`
  - `utils/plotting.py`
- Notebook schema validation for `RL_assisted_MPC_combined_unified.ipynb`
- Shortened `rl-env` smoke runs for:
  - all four agents enabled in nominal mode
  - all four agents enabled in disturbance mode
  - mixed TD3/SAC configuration
  - subset-agent scenarios
