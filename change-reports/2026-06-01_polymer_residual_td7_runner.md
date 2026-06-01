# Polymer Residual TD7 Runner

## Summary

Added a polymer residual TD7/SALE entrypoint beside the existing TD3/SAC residual runner.

## Files Changed

- `RL_assisted_MPC_residual_td7_unified.py`: new TD7-only polymer residual runner using `TD7Agent` and the shared residual supervisor.
- `systems/polymer/notebook_params.py`: added residual TD7 run profiles and a `td7_agent` config block.

## Validation

- Compiled the new runner and modified polymer defaults.
- Verified `get_polymer_notebook_defaults("residual")` still defaults to TD3 and now includes TD7 residual profiles/config.

## Runtime Notes

- The existing `RL_assisted_MPC_residual_unified.py` remains unchanged.
- No polymer training run was launched during this wiring step.
