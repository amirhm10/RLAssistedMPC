# Supervisor-Gated DQN Core

Date: 2026-06-03

## Summary

Added core SG-DQN and SG-dueling-DQN agent support without adding runners, notebook defaults, result prefixes, or experiment configs.

The new discrete gate mirrors the SG-TD3 execution principle for DQN action indices. It compares the learned Q value of a candidate action against a supervisor action and stores replay transitions under the executed action.

## Files Added Or Updated

- `DQN/supervisor_gated_dqn_agent.py`
- `DuelingDQN/supervisor_gated_dueling_dqn_agent.py`
- `DuelingDQN/__init__.py`
- `tests/test_supervisor_gated_dqn.py`

## Behavior

- Added `DiscreteSupervisorGateConfig`.
- Added `DiscreteSupervisorGatedDecision`.
- Added scalar discrete supervisor replay metadata for policy, supervisor, previous, selected source, scores, and advantage.
- Added `SupervisorGatedDQNAgent`.
- Added `SupervisorGatedDuelingDQNAgent`.
- Preserved existing `DQNAgent` and `DuelingDQNAgent` behavior.
- Limited v1 supervised mode to `one_step` and `n_step` with `n_step == 1`.

## Validation

Commands run:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile DQN\supervisor_gated_dqn_agent.py DuelingDQN\supervisor_gated_dueling_dqn_agent.py DuelingDQN\__init__.py tests\test_supervisor_gated_dqn.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_dqn.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_td3.py
```

No Aspen run was launched.
