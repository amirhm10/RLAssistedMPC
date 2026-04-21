# 2026-04-21 Structured Cap Application Hardening

- Hardened [utils/structured_matrix_runner.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py) so the shared structured runner no longer trusts a stale precomputed `structured_spec` when the runtime config carries different `range_profile` or `A/B` override bounds.
- The runner now rebuilds the structured spec when the notebook config and the passed spec do not match, which ensures the intended capped `A`-side bounds are actually applied even if an older notebook cell left `STRUCTURED_SPEC` in memory.
- Added `structured_spec_refreshed` to the saved result bundle so capped structured runs can be verified directly from `input_data.pkl`.
