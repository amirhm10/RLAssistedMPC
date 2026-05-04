# Matrix Decision Intervals

## Summary

Added standalone matrix and structured-matrix decision intervals for polymer and distillation RL-assisted MPC runs.

- Polymer matrix and structured-matrix defaults use `decision_interval = 1`.
- Distillation matrix and structured-matrix defaults use `decision_interval = 20`.
- Held actions only throttle fresh RL action selection. Plant stepping, observer updates, reward calculation, replay insertion, and training still occur every eligible environment step.

## Implementation Notes

- `utils.agent_step_runtime.select_continuous_action` now supports a held continuous action, a decision flag, and a held-action source code.
- `utils.matrix_runner` and `utils.structured_matrix_runner` keep the last raw action and reuse it between decision points.
- The cadence starts from the first live action step after warm start and hidden action freeze.
- A train/test mode switch forces a fresh action so exploratory training actions are not carried into evaluation.

## Verification

- Ran Python syntax checks for the updated shared runners, notebook parameter modules, and dependent continuous-agent runners.
- Ran focused selector checks for warm-start behavior, live action holding, and train/test mode refresh.
- Verified source-level defaults for polymer and distillation matrix families.

## Files Inspected

- `utils/agent_step_runtime.py`
- `utils/matrix_runner.py`
- `utils/structured_matrix_runner.py`
- `utils/phase1_hidden_release.py`
- `systems/polymer/notebook_params.py`
- `systems/distillation/notebook_params.py`
- `RL_assisted_MPC_matrices_unified.ipynb`
- `RL_assisted_MPC_structured_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_structured_matrices_unified.ipynb`

## Files Changed

- `utils/agent_step_runtime.py`
- `utils/matrix_runner.py`
- `utils/structured_matrix_runner.py`
- `utils/phase1_hidden_release.py`
- `systems/polymer/notebook_params.py`
- `systems/distillation/notebook_params.py`
- `utils/polymer_multiseed_core_study.py`
- The four standalone matrix notebook config cells listed above.
