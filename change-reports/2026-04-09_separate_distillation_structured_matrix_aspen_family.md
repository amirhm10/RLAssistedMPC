## Summary

The distillation structured-matrix notebook was sharing the same Aspen family mapping as the original distillation matrix notebook. That meant both notebooks resolved the same `DYN_PATH` / `SNAPS_PATH` and could interfere with each other when run at the same time.

## Changes

- Added dedicated Aspen family mappings in [systems/distillation/config.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/distillation/config.py):
  - `structured_matrix_td3`
  - `structured_matrix_sac`
- Updated [distillation_RL_assisted_MPC_structured_matrices_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_structured_matrices_unified.ipynb) to use the structured family names instead of reusing:
  - `matrix_td3`
  - `matrix_sac`

## Default Aspen mapping

- `structured_matrix_td3`
  - `none` -> `C2S_SS_simulation10.dynf`
  - `ramp` -> `C2S_SS_simulation11.dynf`
  - `fluctuation` -> `C2S_SS_simulation12.dynf`
- `structured_matrix_sac`
  - `none` -> `C2S_SS_simulation13.dynf`
  - `ramp` -> `C2S_SS_simulation13.dynf`
  - `fluctuation` -> `C2S_SS_simulation13.dynf`

## Validation

- `python -m py_compile systems/distillation/config.py`
- notebook code-cell AST parse for the structured distillation notebook
- `resolve_aspen_paths(...)` check in `rl-env` for both structured families
