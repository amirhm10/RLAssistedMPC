# Polymer SG Standard-State Runner Ablation

## Summary

- Switched polymer supervisor-gated horizon, weight, and residual entrypoints from mismatch-state inputs to standard-state inputs.
- Updated output prefixes to include `standard` so new ablation runs are distinguishable from prior mismatch-state runs.
- Left polymer Markov SG wrappers unchanged because the shared Markov runner builds a hardcoded `mismatch_conditioned` Markov state rather than using the normal `state_mode` switch.

## Updated Entrypoints

- `RL_assisted_MPC_horizons_supervisor_gated_dqn_unified.py`
- `RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified.py`
- `RL_assisted_MPC_weights_supervisor_gated_sac_critic_warm_unified.py`
- `RL_assisted_MPC_residual_supervisor_gated_sac_critic_warm_unified.py`
- `RL_assisted_MPC_weights_supervisor_gated_td3_critic_warm_unified.py`
- `RL_assisted_MPC_residual_supervisor_gated_td3_critic_warm_unified.py`
- `RL_assisted_MPC_residual_supervisor_gated_td3_unified.py`

## Validation

- `C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile` on all edited runner files.
- `C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests/test_supervisor_gated_horizon_runners.py`
- `C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests/test_supervisor_gated_sac_runners.py`
- `C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests/test_supervisor_gated_residual_integration.py`

`pytest` was not available in `rl-env`, so the direct test runners were used.
