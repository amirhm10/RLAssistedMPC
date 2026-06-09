# Distillation Combined Runner Reactivation

Date: 2026-06-09

## Summary

Reactivated `distillation_RL_assisted_MPC_combined_unified.py` as an active distillation combined supervisor entrypoint. The runner follows the polymer combined style with horizon, Markov, weights, and residual sub-agents, while keeping the legacy matrix branch disabled.

## Implementation Notes

- Added `resolve_distillation_combined_agent_kinds(mode)` with `sg` and `plain`/`without_sg`/`no_sg` mappings.
- Reintroduced the `combined` distillation notebook defaults.
- Built active combined defaults from the active standalone horizon, Markov, weights, and residual defaults.
- Kept combined matrix disabled and guarded with an early runner error if enabled.
- Kept residual rho authority and residual deadband disabled by default, matching the active residual defaults.
- Instantiated only DQN, SG-DQN, TD3, and SG-TD3 agents in the active combined runner.
- Added simple future result prefixes such as `distillation_combined_sg_disturb_fluctuation`.
- Removed the stale pre-reactivation combined default block so only the standalone-derived defaults define active combined behavior.
- Fixed the disabled matrix placeholder bounds to strictly bracket nominal multiplier 1.0, because the shared combined runtime maps the matrix baseline action even when the matrix agent is disabled.

## Verification

- Python syntax checks for modified distillation default modules, active runner, and tests.
- `tests/test_distillation_combined_runner.py`
- `tests/test_distillation_supervisor_gated_sac_runners.py`
- `tests/test_supervisor_gated_horizon_runners.py`
- `tests/test_supervisor_gated_residual_integration.py`
- `tests/test_exploration_defaults.py`
- `tests/test_polymer_combined_runner.py`

Existing timestamped results and generated data directories were not renamed or regenerated.
