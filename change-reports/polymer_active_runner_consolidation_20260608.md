# Polymer Active Runner Consolidation - 2026-06-08

## Active Polymer Entrypoints

Polymer RL execution is now centered on four root runners:

- `RL_assisted_MPC_horizons_unified.py`
- `RL_assisted_MPC_weights_unified.py`
- `RL_assisted_MPC_residual_unified.py`
- `RL_assisted_MPC_markov_unified.py`

The active defaults are supervisor-gated and disturbance-mode:

- Horizons: `agent_kind="sg_dqn"`, selectable plain mode `agent_kind="dqn"`.
- Weights, residual, and Markov: `agent_kind="sg_td3"`, selectable plain mode `agent_kind="td3"`.
- All active defaults use `run_mode="disturb"` and `state_mode="mismatch"`.

## Default SG Settings

- SG-DQN horizons use the OF-MPC/default horizon supervisor, warm start `10`, post-warm action freeze `3`, one-step replay, epsilon `0.2 -> 0.02`, and gate margin `0.0`.
- SG-TD3 weights use the identity supervisor, warm start `10`, action freeze `3`, actor freeze `3`, SG margin `0.5`, uncertainty penalty `0.5`, supervisor-action penalty `0.05`, previous-action penalty `0.01`, and nonfinite identity fallback.
- SG-TD3 residual uses the zero-residual supervisor, warm start `10`, action freeze `3`, actor freeze `3`, SG margin `0.5`, uncertainty penalty `0.5`, supervisor-action penalty `0.05`, previous-action penalty `0.01`, and nonfinite zero fallback.
- SG-TD3 Markov uses the LS-or-MPC supervisor, warm start `10`, action freeze `3`, actor freeze `3`, SG margin `0.0`, uncertainty penalty `0.5`, supervisor-action penalty `0.02`, previous-action penalty `0.01`, live Markov safety disabled, and shadow diagnostics retained.

The older report recommendation that standard-state mode could be the default is intentionally superseded for active polymer SG runs. Mismatch-state mode is the current default so the next polymer reruns test the richer state information consistently.

## Archived Entrypoints

Inactive polymer algorithm-specific root entrypoints were moved to:

`archive/polymer_inactive_algorithm_entrypoints_20260608/`

This includes polymer SAC/SG-SAC wrappers, dueling and SG-dueling horizon entrypoints, TD7 residual, combined supervisor, old SG-TD3 wrappers, and standard-standard Markov ablations.

Reusable agent packages were not removed or archived. `SACAgent/`, `DuelingDQN/`, `TD7Agent/`, and shared utility support remain available for historical analysis, distillation compatibility, and future ablations.
