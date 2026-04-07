# Repo-Wide RL Exploration, DDQN Cleanup, Dueling Alignment, And Diagnostics

## What changed
- Refactored the standard discrete horizon stack under `DQN/` into a clean single-online/single-target Double DQN.
- Expanded `DuelingDQN/` into a full dueling DDQN path with optional NoisyNet exploration and richer diagnostics.
- Added shared factorized-Gaussian NoisyNet layers in `utils/noisy_layers.py`.
- Added the missing distillation dueling horizon notebook:
  - `distillation_RL_assisted_MPC_horizons_dueling_unified.ipynb`
- Extended TD3 with configurable critic loss and parameter-noise exploration.
- Extended SAC with configurable critic loss and richer diagnostics.
- Upgraded mismatch-state normalization to use explicit per-output supervisory scales and configurable clipping.
- Expanded result bundles and plotting so saved `input_data.pkl` bundles can drive richer diagnostics and simple multi-run summaries later.

## Touched files
- `DQN/qnetwork.py`
- `DQN/dqn_agent.py`
- `DuelingDQN/qnetwork.py`
- `DuelingDQN/dueling_dqn_agent.py`
- `TD3Agent/agent.py`
- `SACAgent/sac_agent.py`
- `utils/noisy_layers.py`
- `utils/state_features.py`
- `utils/horizon_runner.py`
- `utils/horizon_runner_dueling.py`
- `utils/matrix_runner.py`
- `utils/weights_runner.py`
- `utils/residual_runner.py`
- `utils/combined_runner.py`
- `utils/plotting.py`
- `utils/plotting_core.py`
- `RL_assisted_MPC_horizons_unified.ipynb`
- `RL_assisted_MPC_horizons_dueling_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_dueling_unified.ipynb`
- `RL_assisted_MPC_matrices_unified.ipynb`
- `RL_assisted_MPC_weights_unified.ipynb`
- `RL_assisted_MPC_residual_unified.ipynb`
- `RL_assisted_MPC_combined_unified.ipynb`
- `distillation_RL_assisted_MPC_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_weights_unified.ipynb`
- `distillation_RL_assisted_MPC_residual_unified.ipynb`
- `distillation_RL_assisted_MPC_combined_unified.ipynb`
- `report/dueling_double_dqn_horizon_supervisor.tex`

## New defaults
- Standard horizon notebooks:
  - `exploration_mode = "epsilon"`
  - `loss_type = "huber"`
- Dueling horizon notebooks:
  - `exploration_mode = "noisy"`
  - `loss_type = "huber"`
- TD3 research-facing unified notebooks:
  - `exploration_mode = "param_noise"`
  - `loss_type = "huber"`
- SAC unified notebooks:
  - `loss_type = "huber"`
- Mismatch mode:
  - explicit `mismatch_scale`
  - `mismatch_clip = 3.0`

## New notebook entrypoints
- `RL_assisted_MPC_horizons_dueling_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_dueling_unified.ipynb`

## Reproduce
1. Use the canonical unified notebooks under the repo root.
2. For polymer, use `Polymer/Data` and `Polymer/Results`.
3. For distillation, use `Distillation/Data` and `Distillation/Results`.
4. Standard horizon notebooks use the cleaned DDQN baseline; dueling notebooks use the NoisyNet-ready dueling DDQN path.
5. Saved bundles can be compared later with the plotting layer directly from `input_data.pkl`.
