# N-Step Credit Assignment Extension

## Summary
- Added plain uncorrected n-step return support across the active RL stacks:
  - `DQN/`
  - `DuelingDQN/`
  - `TD3Agent/`
  - `SACAgent/`
- Kept `n_step = 1` as the default baseline everywhere.
- Exposed visible `N_STEP` controls in the active polymer/distillation unified notebooks for:
  - standard horizon
  - dueling horizon
  - matrices
  - weights
  - residual
- Extended result bundles and plotting so saved `input_data.pkl` bundles can carry n-step metadata and diagnostics without breaking older bundles.

## Main Files
- `utils/nstep.py`
- `DQN/replay_buffer.py`
- `TD3Agent/replay_buffer.py`
- `DQN/dqn_agent.py`
- `DuelingDQN/dueling_dqn_agent.py`
- `TD3Agent/agent.py`
- `SACAgent/sac_agent.py`
- `utils/horizon_runner.py`
- `utils/horizon_runner_dueling.py`
- `utils/matrix_runner.py`
- `utils/weights_runner.py`
- `utils/residual_runner.py`
- `utils/plotting_core.py`
- `utils/plotting.py`
- `systems/polymer/notebook_params.py`
- `systems/distillation/notebook_params.py`
- active polymer/distillation unified notebooks for horizon, dueling horizon, matrices, weights, and residual
- `report/dueling_double_dqn_horizon_supervisor.tex`

## Behavior
- Agents still receive one-step `push(s, a, r, ns, done)` calls from the runners.
- Each agent now aggregates n-step endpoint transitions internally before replay insertion.
- DDQN / dueling DDQN use online `argmax` plus target-network evaluation at the endpoint.
- TD3 and SAC keep their existing target constructions, but now consume aggregated reward/discount values.
- `n_step = 1` reduces to the original one-step behavior.

## Notebook Defaults
- The system parameter modules now expose `n_step` in the notebook-facing defaults.
- The edited unified notebooks now:
  - define visible `N_STEP` locals,
  - pass `n_step` into agent constructors,
  - include `n_step` in grouped summaries,
  - include `n_step` in the saved config snapshots (`horizon_cfg`, `dueling_cfg`, `matrix_cfg`, `weight_cfg`, `residual_cfg`).

## Saved Bundle / Plotting Additions
- Added backward-compatible support for:
  - `n_step`
  - `reward_n_mean_trace`
  - `discount_n_mean_trace`
  - `bootstrap_q_mean_trace`
  - `n_actual_mean_trace`
  - `truncated_fraction_trace`
- Horizon, matrix, weight, and residual plotting now emit n-step decomposition figures when those traces are present.

## Validation
- Python compile checks on the touched replay, agent, runner, and plotting modules.
- Notebook JSON / code-cell parse checks for the touched unified notebooks.
- Targeted sanity checks for:
  - `NStepAccumulator`
  - replay storage/sampling of aggregated transitions
  - DDQN / dueling / TD3 / SAC constructor compatibility with `n_step`
  - plotting normalization on the new n-step traces
