## Distillation Markov Single-Session Aspen Fix

- Updated `distillation_RL_assisted_MPC_markov_unified.ipynb` to follow the same pattern as the other distillation notebooks: open one Aspen system, extract steady-state metadata from that live session, and pass the same system into the shared runner.
- Updated `utils/markov_runner.py` so the shared Markov runner accepts a live `runtime_ctx["system"]` and no longer calls `system_factory()` during runtime-context construction just to infer `delta_t`.
- This prevents the distillation Markov workflow from opening a probe Aspen file, reaching steady state, and then opening a second Aspen session before the actual rollout begins.
