# SAC Corrections and SG-SAC Core

## Summary

- Corrected the SAC core to use a configurable actor Q mode, defaulting to conservative `min(Q1, Q2)`.
- Switched adaptive entropy tuning to the standard log-temperature objective and froze alpha during critic-warm actor freeze by default.
- Corrected tanh-Gaussian log probabilities for `max_action != 1`.
- Added a core `SupervisorGatedSACAgent` with SG-TD3-style candidate-vs-supervisor gating, supervised replay metadata, and optional supervisor actor regularization.

## Scope

- No SAC, SG-SAC, polymer, or distillation runners were executed.
- No SG-SAC runner entrypoints, notebook defaults, result prefixes, or experiment configs were added.
- Existing SAC runner behavior remains stochastic in training and deterministic in evaluation.

## Validation

- `C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile SACAgent\sac_agent.py SACAgent\gaussian_actor.py SACAgent\supervisor_gated_sac_agent.py tests\test_sac_agent_core.py tests\test_supervisor_gated_sac.py`
- `C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_sac_agent_core.py`
- `C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_sac.py`
- `C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests\test_supervisor_gated_td3.py`
