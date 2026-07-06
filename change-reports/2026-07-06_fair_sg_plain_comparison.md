# Fair SG vs Plain Comparison Alignment

## Context

The SG and plain active runners are intended to compare the effect of the
supervisory gate only. Warm-start behavior, post-warm action freeze, actor
freeze, numerical nonfinite fallback, solver-failure fallback, action bounds,
and disabled safety-layer defaults should be identical between modes.

## Change

- Shared the single Markov post-warm action-freeze and actor-freeze behavior
  across plain TD3 and SG-TD3/SG-SAC Markov runs.
- Made plain Markov use the same LS-else-nominal supervisor action as SG during
  warm/post-warm freeze and nonfinite fallback.
- Kept `force_td3_execute=True` as direct TD3 execution only when the corrected
  MPC candidate solve succeeds.
- Aligned combined Markov so forced TD3 execution still falls back to the same
  supervisor action on corrected-candidate solver failure.
- Added the missing polymer combined pass-through for
  `force_td3_respects_warm_start` so combined uses the same Markov controller
  contract as the single runner.

## Validation

- In-memory syntax compile passed for `utils/markov_runner.py`,
  `utils/combined_runner.py`, `utils/agent_step_runtime.py`,
  `systems/polymer/notebook_params.py`, and
  `RL_assisted_MPC_combined_unified.py`.
- Config audit confirmed polymer and distillation Markov/combined defaults use
  plain mode, shared 3-subepisode action/actor freeze, `force_td3_execute=True`,
  `force_td3_respects_warm_start=True`, `rl_fallback_to_ls=False`, and disabled
  live `z_safety` / TD3-priority fallback.
- `py_compile` was attempted first, but the local `utils/__pycache__` directory
  denied `.pyc` replacement writes.
