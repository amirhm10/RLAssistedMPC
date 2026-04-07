# Add Dueling DDQN Horizon Supervisor

## Summary
- Added a new parallel discrete RL package under `DuelingDQN/`.
- Added a new polymer notebook, `RL_assisted_MPC_horizons_dueling_unified.ipynb`.
- Added a new shared runner, `utils/horizon_runner_dueling.py`.
- Extended horizon plotting to show optional dueling-DDQN diagnostics without breaking existing result bundles.
- Added a LaTeX technical note under `report/`.

## Intentionally Unchanged
- The legacy `DQN/` package remains the baseline discrete stack.
- The existing `RL_assisted_MPC_horizons_unified.ipynb` still uses the legacy `DQNAgent`.
- The current shared horizon runner in `utils/horizon_runner.py` was left intact.

## Notes
- The new dueling path reuses the current PER replay buffer implementation through a small shim in `DuelingDQN/replay_buffer.py`.
- The new implementation uses a standard online/target dueling-DDQN design instead of the legacy twin-discrete-Q structure, because the current legacy horizon workflow is effectively `q1`-driven.
