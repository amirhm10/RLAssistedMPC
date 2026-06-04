# Polymer Markov Standard-Standard Ablation

Date: 2026-06-04

## Summary

Added a base-only Markov RL observation option for polymer Markov SG runs. The existing Markov `standard` mode still feeds Markov-specific features to the agent:

- previous correction `z_prev`
- LS correction `z_ls`
- LS prediction score
- LS gain drift

The new `markov_agent_state_features = "base_only"` option removes those Markov-specific features and feeds only the standard base RL state to the policy.

## New Entrypoints

- `RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_standard_standard_unified.py`
- `RL_assisted_MPC_markov_supervisor_gated_sac_critic_warm_standard_standard_unified.py`

Both runners keep the same LS-or-MPC supervisor and shadow-only safety settings as the existing Markov SG runners, but write result prefixes with `standard_standard`.

## Validation

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile utils\markov_runner.py RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_standard_standard_unified.py RL_assisted_MPC_markov_supervisor_gated_sac_critic_warm_standard_standard_unified.py tests\test_supervisor_gated_markov_integration.py tests\test_supervisor_gated_sac_runners.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_markov_integration.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_sac_runners.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_td3.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_sac.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_sac_agent_core.py
```
