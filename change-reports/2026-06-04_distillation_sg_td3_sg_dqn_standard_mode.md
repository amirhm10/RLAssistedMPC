# Distillation SG-TD3 and SG-DQN Standard-State Switch

Date: 2026-06-04

## Summary

Changed the distillation supervisor-gated TD3 and DQN experiment wrappers from mismatch-state observations to standard observations for the next ablation run set.

Updated families:

- SG-TD3 weights
- SG-TD3 residual
- SG-TD3 Markov
- SG-DQN horizon
- SG-dueling-DQN horizon
- SG-dueling-DQN Aspen-6 legacy-reward horizon check

The wrappers still default to `run_mode = "disturb"` and `disturbance_profile = "fluctuation"`. Result and compare prefixes now include `standard` so the new runs remain distinguishable from previous mismatch-mode runs.

## Validation

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile distillation_RL_assisted_MPC_weights_supervisor_gated_td3_critic_warm_unified.py distillation_RL_assisted_MPC_residual_supervisor_gated_td3_critic_warm_unified.py distillation_RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_unified.py distillation_RL_assisted_MPC_horizons_supervisor_gated_dqn_unified.py distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified.py distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_aspen6_legacy_reward_unified.py tests\test_supervisor_gated_horizon_runners.py tests\test_supervisor_gated_residual_integration.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_horizon_runners.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_residual_integration.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_td3.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_dqn.py
```
