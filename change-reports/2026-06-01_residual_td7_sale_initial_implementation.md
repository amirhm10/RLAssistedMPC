# Residual TD7/SALE Initial Implementation

## Summary

Implemented a residual-only TD7/SALE path without replacing the existing TD3 or SAC residual workflows.

## Files Changed

- `TD7Agent/`: new TD7 agent package with SALE encoders, actor, twin critic, Hybrid LAP replay, checkpoint/fixed/target networks, and TD3-compatible runtime API.
- `distillation_RL_assisted_MPC_residual_td7_unified.py`: new dedicated residual TD7 entrypoint using the existing distillation residual family.
- `systems/distillation/config.py`: added TD7 residual run profiles copied from TD3 residual profiles.
- `systems/distillation/notebook_params.py`: added residual `td7_agent` defaults while preserving default `agent_kind = "td3"`.
- `utils/residual_runner.py`: allowed `agent_kind = "td7"` and attached TD7 traces to result bundles.

## Validation

- `py_compile` passed for all new TD7 files, modified config/runner files, and the new residual TD7 entrypoint.
- Verified `get_distillation_notebook_defaults("residual")` still defaults to TD3 and includes TD7 profiles.
- Ran a CPU-only TD7 smoke test with random transitions, tensor-shape checks, Hybrid LAP priority updates, BC diagnostics, finite TD7 losses/traces, replay snapshot export, and save/load.

## Runtime Notes

- No Aspen/distillation runtime was launched.
- The TD7 runner uses family `residual` for Aspen path resolution, so it does not add a new distillation family or simulation number.
- Legacy `td3_authority_ramp` naming is preserved for bundle and plotting compatibility.
