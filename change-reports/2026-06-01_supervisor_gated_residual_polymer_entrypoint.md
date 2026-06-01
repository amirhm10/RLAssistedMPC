# Supervisor-Gated Polymer Residual Entrypoint

## Summary

- Updated `AGENTS.md` to use this machine's `rl-env` interpreter at `C:\Users\hamediaa\.conda\envs\rl-env\python.exe`.
- Added `sg_td3` as an opt-in polymer residual profile with nominal and disturbance result prefixes.
- Added `RL_assisted_MPC_residual_supervisor_gated_td3_unified.py`, a dedicated polymer residual entrypoint that instantiates `SupervisorGatedTD3Agent`.
- Extended `utils/residual_runner.py` so `sg_td3` uses zero residual as the supervisor action, logs gate diagnostics, and stores supervisor metadata with the executed residual action.

## Scientific Guardrails

- Existing TD3, SAC, and TD7 residual entrypoints remain unchanged.
- The supervisor-gated path still trains the critic only on the final executed residual action.
- No distillation or Aspen execution is part of this change.

## Validation

- Compile and smoke tests should be run with `C:\Users\hamediaa\.conda\envs\rl-env\python.exe`.
- Added a lightweight integration test for the `sg_td3` residual config and runner import.
