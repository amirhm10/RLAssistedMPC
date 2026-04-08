# Advanced Multistep Credit Assignment Extension

## Summary
- Extended the current plain `n_step` infrastructure into a Stage 1 advanced multistep layer.
- Added shared sequence sampling, truncated lambda-return targets, discrete Retrace for epsilon-greedy DDQN/dueling DDQN, and SAC-n support.
- Kept `n_step = 1` and `multistep_mode = "one_step"` as the default behavior across the repo.
- Left combined notebooks unchanged in this pass.

## Main Code Changes

### Shared utilities
- Added [utils/sequence_sampling.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/sequence_sampling.py)
  - contiguous replay-sequence sampling
  - explicit sequence masks and realized lengths
  - safe handling of episode boundaries and ring-buffer ordering
- Added [utils/nstep_targets.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/nstep_targets.py)
  - endpoint target helpers
  - truncated forward-view lambda returns
  - discrete Retrace target builder
  - SAC-n endpoint soft target helper

### Replay extensions
- Updated [DQN/replay_buffer.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/DQN/replay_buffer.py)
- Updated [TD3Agent/replay_buffer.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/TD3Agent/replay_buffer.py)

Both replay families now support:
- standard transition sampling for `one_step` and `n_step`
- optional contiguous sequence sampling for advanced multistep modes
- episode metadata needed to avoid crossing resets/terminals
- discrete behavior-probability metadata for exact epsilon-greedy Retrace

### Agent changes
- Updated [DQN/dqn_agent.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/DQN/dqn_agent.py)
- Updated [DuelingDQN/dueling_dqn_agent.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/DuelingDQN/dueling_dqn_agent.py)
- Updated [TD3Agent/agent.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/TD3Agent/agent.py)
- Updated [SACAgent/sac_agent.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/SACAgent/sac_agent.py)

New visible agent-level controls:
- `n_step`
- `multistep_mode`
- `lambda_value`

Supported Stage 1 mode matrix:
- DDQN / dueling DDQN:
  - `"one_step"`
  - `"n_step"`
  - `"lambda"`
  - `"retrace"`
- TD3:
  - `"one_step"`
  - `"n_step"`
  - `"lambda"`
- SAC:
  - `"one_step"`
  - `"n_step"`
  - `"sac_n"`
  - `"lambda"`

Important constraint:
- exact discrete Retrace is allowed only with `exploration_mode="epsilon"`
- Retrace + NoisyNet now raises a clear configuration error

### Runner and bundle updates
- Updated:
  - [utils/horizon_runner.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/horizon_runner.py)
  - [utils/horizon_runner_dueling.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/horizon_runner_dueling.py)
  - [utils/matrix_runner.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py)
  - [utils/weights_runner.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/weights_runner.py)
  - [utils/residual_runner.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/residual_runner.py)

Saved bundles now include advanced multistep metadata and diagnostics when present:
- `multistep_mode`
- `lambda_value`
- `reward_n_mean_trace`
- `discount_n_mean_trace`
- `bootstrap_q_mean_trace`
- `n_actual_mean_trace`
- `truncated_fraction_trace`
- `lambda_return_mean_trace`
- `offpolicy_rho_mean_trace`
- `offpolicy_c_mean_trace`
- `behavior_logprob_mean_trace`
- `target_logprob_mean_trace`
- `retrace_c_clip_fraction_trace`

### Notebook/config integration
- Updated notebook defaults in:
  - [systems/polymer/notebook_params.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/polymer/notebook_params.py)
  - [systems/distillation/notebook_params.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/distillation/notebook_params.py)

- Updated active unified notebooks:
  - [RL_assisted_MPC_horizons_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_horizons_unified.ipynb)
  - [RL_assisted_MPC_horizons_dueling_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_horizons_dueling_unified.ipynb)
  - [RL_assisted_MPC_matrices_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_matrices_unified.ipynb)
  - [RL_assisted_MPC_weights_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_weights_unified.ipynb)
  - [RL_assisted_MPC_residual_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_residual_unified.ipynb)
  - [distillation_RL_assisted_MPC_horizons_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_horizons_unified.ipynb)
  - [distillation_RL_assisted_MPC_horizons_dueling_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_horizons_dueling_unified.ipynb)
  - [distillation_RL_assisted_MPC_matrices_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_matrices_unified.ipynb)
  - [distillation_RL_assisted_MPC_weights_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_weights_unified.ipynb)
  - [distillation_RL_assisted_MPC_residual_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_residual_unified.ipynb)

Visible notebook variables now include:
- `N_STEP`
- `MULTISTEP_MODE`
- `LAMBDA_VALUE`

These values are shown in the grouped notebook summaries and passed into the saved config snapshots.

### Plotting/reporting
- Updated [utils/plotting_core.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/plotting_core.py)
- Updated [utils/plotting.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/plotting.py)
- Updated [report/dueling_double_dqn_horizon_supervisor.tex](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/report/dueling_double_dqn_horizon_supervisor.tex)

Plotting remains backward-compatible with old bundles and now adds advanced multistep diagnostics when present.

## Defaults
- `n_step = 1` remains the repo-wide baseline default
- `multistep_mode = "one_step"` remains the repo-wide baseline default
- dueling horizon notebook defaults still keep the research-facing `multistep_mode = "n_step"`
- `lambda_value = 0.9` is the visible research default when lambda-style modes are selected

## Validation
- `py_compile` passed for the touched shared utilities, replay buffers, agents, runners, plotting modules, and notebook param modules
- `nbformat` + `ast` parsing passed for the touched unified notebooks
- `rl-env` sanity checks passed for:
  - sequence sampling staying inside episode boundaries
  - truncated lambda-return numerical finiteness
  - Retrace rejection under NoisyNet exploration
  - DDQN synthetic `n_step` training
  - dueling DDQN synthetic `lambda` training
  - TD3 synthetic `lambda` training
  - SAC synthetic `sac_n` training
  - SAC synthetic `lambda` training

## Deferred
- V-trace is intentionally not implemented in this Stage 1 pass
- combined notebooks remain out of scope for multistep training logic in this pass
