## Summary

Cleaned the active TD3/SAC replay path so the repo uses one mixed PER/recent replay buffer in the online continuous workflow, removed active pretraining hooks, fixed twin-critic priority updates, added end-of-rollout n-step flushing, and aligned notebook-facing replay defaults across the active notebook families.

## Files Changed

- `TD3Agent/agent.py`
- `TD3Agent/replay_buffer.py`
- `SACAgent/sac_agent.py`
- `DQN/dqn_agent.py`
- `DQN/replay_buffer.py`
- `DuelingDQN/dueling_dqn_agent.py`
- `utils/matrix_runner.py`
- `utils/weights_runner.py`
- `utils/residual_runner.py`
- `utils/combined_runner.py`
- `systems/polymer/notebook_params.py`
- `systems/distillation/notebook_params.py`
- `RL_assisted_MPC_horizons_unified.ipynb`
- `RL_assisted_MPC_horizons_dueling_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_dueling_unified.ipynb`
- `RL_assisted_MPC_matrices_unified.ipynb`
- `RL_assisted_MPC_weights_unified.ipynb`
- `RL_assisted_MPC_residual_unified.ipynb`
- `distillation_RL_assisted_MPC_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_weights_unified.ipynb`
- `distillation_RL_assisted_MPC_residual_unified.ipynb`
- `RL_assisted_MPC_combined_unified.ipynb`
- `distillation_RL_assisted_MPC_combined_unified.ipynb`

## What Changed

### Active TD3/SAC replay cleanup

- Removed the active plain-vs-mixed replay selection branches.
- TD3 and SAC now always instantiate `PERRecentReplayBuffer` in the active codepath.
- Removed active pretraining-only hooks from the main TD3/SAC agents.
- Added explicit replay constructor knobs for the mixed buffer:
  - `buffer_size`
  - `replay_frac_per`
  - `replay_frac_recent`
  - `replay_recent_window`
  - `replay_alpha`
  - `replay_beta_start`
  - `replay_beta_end`
  - `replay_beta_steps`

### Priority updates

- TD3 and SAC now update PER priorities using both critics:
  - `0.5 * (abs(y - q1) + abs(y - q2))`

### End-of-rollout n-step flush

- Added `flush_nstep()` to the active TD3 and SAC agents.
- Continuous runners now call the flush at rollout end so the final partial n-step transitions are not dropped:
  - matrix
  - weights
  - residual
  - combined

### Replay buffer stateful defaults

- The mixed replay buffer and the discrete PER/recent buffer now store their sampling knobs on the buffer object.
- `sample(...)` and `sample_sequence(...)` use the stored defaults unless a test overrides them explicitly.

### Notebook-facing default alignment

- Unified replay defaults are now exposed from the polymer/distillation notebook default modules.
- Active notebooks now show a consistent replay parameter surface where applicable:
  - `buffer_size = 40000`
  - `replay_frac_per = 0.5`
  - `replay_frac_recent = 0.2`
  - `replay_recent_window_mult = 5`
  - `replay_recent_window`
  - `replay_alpha = 0.6`
  - `replay_beta_start = 0.4`
  - `replay_beta_end = 1.0`
  - `replay_beta_steps = 50000`
- Continuous notebooks derive the effective recent window as:
  - `min(buffer_size, replay_recent_window_mult * set_points_len)`
- Grouped notebook summaries now print resolved replay settings.

## Out Of Scope

- `TD3Agent/agent_modified.py`
- `TD3Agent/replay_buffer_modified.py`

These legacy alternate files were intentionally left untouched.

## Validation

- `py_compile` passed for the touched Python modules.
- The touched notebooks parsed successfully with `nbformat`/`ast`.
- Search checks confirmed the active TD3/SAC path no longer contains the removed replay/pretraining branches.
- A short `rl-env` sanity check confirmed TD3 and SAC still instantiate on `PERRecentReplayBuffer` and `flush_nstep()` preserves pending multistep transitions.
