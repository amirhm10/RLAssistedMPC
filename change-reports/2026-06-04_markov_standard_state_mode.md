# Markov Standard-State Mode

## Summary

- Added a real `state_mode` switch to the shared Markov runner.
- `state_mode = "standard"` now builds the Markov RL observation from the standard partial observation only, then appends the Markov context:
  - conditioned base state from `x`, setpoint, and previous input
  - previous Markov correction `z`
  - LS-or-MPC supervisor candidate `z`
  - LS score and drift diagnostics
- `state_mode = "mismatch"` keeps the previous mismatch-conditioned Markov state with innovation and tracking-error features.
- Polymer SG-TD3 and SG-SAC Markov wrappers now set `state_mode = "standard"` and include `standard` in result/compare prefixes.
- Distillation Markov unified runner now passes `state_mode` through when configured, but its default behavior remains mismatch unless a wrapper changes it.

## Validation

- `C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile` on edited Markov runner, wrappers, defaults, and tests.
- `C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests/test_supervisor_gated_markov_integration.py`
- `C:\Users\hamediaa\.conda\envs\rl-env\python.exe tests/test_supervisor_gated_sac_runners.py`
