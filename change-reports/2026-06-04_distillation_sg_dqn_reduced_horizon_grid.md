# Distillation SG-DQN Reduced Horizon Grid

Date: 2026-06-04

## Summary

Changed the distillation supervisor-gated horizon DQN wrappers to use a reduced standard-state horizon grid:

- prediction horizon `Np = 6..11`
- control horizon `Nc = 3..11`
- valid recipes keep `Nc <= Np`

This gives `39` discrete actions. The grid keeps the OF-MPC supervisor anchor `(6, 3)` and medium/long candidates such as `(10, 8)` and `(11, 11)`, while removing the short-control `Nc = 2` candidates and the wider high-churn tails.

Updated wrappers:

- `distillation_RL_assisted_MPC_horizons_supervisor_gated_dqn_unified.py`
- `distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified.py`
- `distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_aspen6_legacy_reward_unified.py`

The result and compare prefixes include `np6_11_nc3_11` so these runs remain distinguishable from the previous `87`-action grid.

## Validation

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile distillation_RL_assisted_MPC_horizons_supervisor_gated_dqn_unified.py distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified.py distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_aspen6_legacy_reward_unified.py tests\test_supervisor_gated_horizon_runners.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_horizon_runners.py
```
