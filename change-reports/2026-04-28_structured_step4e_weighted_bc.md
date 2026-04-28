# 2026-04-28 Structured Step 4E weighted BC

## Summary

Implemented the first Step 4E structured-matrix behavioral-cloning refinement:

- optional per-coordinate BC weights in the shared BC helper
- weighted BC penalty support in both TD3 and SAC actor updates
- polymer structured-matrix defaults now use label-based A/B-split BC weights

## Implementation details

- `utils/behavioral_cloning.py`
  - accepts optional `coordinate_weights`
  - accepts optional `label_weight_overrides`
  - resolves a per-coordinate BC weight vector into the runtime `bc_context`
- `TD3Agent/agent.py`
  - applies a weighted squared-error BC penalty when `coordinate_weights` are present
- `SACAgent/sac_agent.py`
  - applies the same weighted BC penalty to the deterministic policy mean
- `utils/structured_matrix_runner.py`
  - passes structured action labels into BC-context resolution
- `systems/polymer/notebook_params.py`
  - polymer structured-matrix BC now uses label-based weights:
    - `A_block_1: 1.25`
    - `A_block_2: 1.0`
    - `A_block_3: 1.0`
    - `A_off: 1.25`
    - `B_col_1: 2.5`
    - `B_col_2: 2.5`
- `systems/distillation/notebook_params.py`
  - keeps the BC config surface symmetric by exposing the new optional keys while remaining disabled

## Validation

- `py_compile` passed for all touched Python files
- polymer structured defaults load with the new label-based BC weights
- shared BC helper resolves the expected per-coordinate weight vector for the current structured action labels

## Intended experiment

This change is the code path for the next polymer structured-matrix Step 4E rerun:

- keep Step 2 and Step 3 off
- keep stronger structured BC schedule
- replace the old uniform BC penalty with the new A/B-split weighted BC penalty
