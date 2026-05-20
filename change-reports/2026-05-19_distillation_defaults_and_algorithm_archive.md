# Distillation Defaults And Algorithm Archive

Date: 2026-05-19

## Summary

- Restored distillation RL defaults to the smaller historical network setting:
  `gamma = 0.99` and hidden layers `[128, 128]`.
- Applied the default through shared constants in `systems/distillation/notebook_params.py` so horizon, dueling horizon, weights, residual, Markov, matrix, structured-matrix, reidentification, and combined family configs stay aligned.
- Updated the Markov runner fallback gamma from `0.995` to `0.99` for defensive consistency when a config omits `td3_agent.gamma`.
- Archived inactive root-level distillation RL algorithm notebooks under `archive/distillation_inactive_algorithm_entrypoints_20260519/`.

## Archived Files

- `distillation_RL_assisted_MPC_combined_unified.ipynb`
- `distillation_RL_assisted_MPC_markov_td3_only_no_safeguard_unified.ipynb`
- `distillation_RL_assisted_MPC_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_reidentification_unified.ipynb`
- `distillation_RL_assisted_MPC_structured_matrices_unified.ipynb`

## Validation

- Imported all distillation notebook-family defaults and confirmed DQN/SAC/TD3 agent defaults resolve to `gamma = 0.99` and `[128, 128]`.
- Ran `py_compile` on `systems/distillation/notebook_params.py` and `utils/markov_runner.py`.
