# Distillation SG-DQN Horizon Runners

Date: 2026-06-03

## Summary

Added two thin distillation horizon runner entrypoints for the new supervisor-gated discrete agents:

- `distillation_RL_assisted_MPC_horizons_supervisor_gated_dqn_unified.py`
- `distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified.py`

Both wrappers reuse the existing standard and dueling horizon runners. They pass in the SG-DQN agent class through a lightweight hook rather than duplicating the full distillation setup.

## Behavior

- Uses the OF-MPC default horizon `(Hp, Hc) = (6, 3)` as the supervisor action.
- Keeps the old `horizon_safety` release filter, reward probation, and shadow diagnostics disabled for the first SG-DQN ablation.
- Preserves the existing high-level horizon action-source log while adding SG-specific policy, supervisor, executed, previous, score, Q, and selected-source logs.
- Trains SG-DQN replay on the final executed action using `push_supervised`.
- Reuses existing Aspen families:
  - SG-DQN uses `horizon`.
  - SG-dueling-DQN uses `horizon_dueling`.

## Files Added Or Updated

- `distillation_RL_assisted_MPC_horizons_supervisor_gated_dqn_unified.py`
- `distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified.py`
- `distillation_RL_assisted_MPC_horizons_unified.py`
- `distillation_RL_assisted_MPC_horizons_dueling_unified.py`
- `utils/agent_step_runtime.py`
- `utils/horizon_runner.py`
- `utils/horizon_runner_dueling.py`
- `tests/test_supervisor_gated_horizon_runners.py`

## Validation

Commands run:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile DQN\supervisor_gated_dqn_agent.py DuelingDQN\supervisor_gated_dueling_dqn_agent.py utils\agent_step_runtime.py utils\horizon_runner.py utils\horizon_runner_dueling.py distillation_RL_assisted_MPC_horizons_unified.py distillation_RL_assisted_MPC_horizons_dueling_unified.py distillation_RL_assisted_MPC_horizons_supervisor_gated_dqn_unified.py distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified.py tests\test_supervisor_gated_horizon_runners.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_dqn.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_horizon_runners.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_td3.py
```

No distillation runner was executed, so no Aspen run was launched.
