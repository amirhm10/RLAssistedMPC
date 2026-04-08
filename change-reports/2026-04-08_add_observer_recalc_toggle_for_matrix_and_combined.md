# Add Observer Recalculation Toggle For Matrix And Combined

## Summary
- Added a notebook-visible toggle to the matrix and combined unified workflows to optionally recompute the observer gain `L` after matrix updates.
- The new toggle defaults to `False` so the legacy observer behavior remains unchanged unless explicitly enabled.

## Files Changed
- [utils/matrix_runner.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py)
- [utils/combined_runner.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/combined_runner.py)
- [systems/polymer/notebook_params.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/polymer/notebook_params.py)
- [systems/distillation/notebook_params.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/distillation/notebook_params.py)
- [RL_assisted_MPC_matrices_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_matrices_unified.ipynb)
- [distillation_RL_assisted_MPC_matrices_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_matrices_unified.ipynb)
- [RL_assisted_MPC_combined_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_combined_unified.ipynb)
- [distillation_RL_assisted_MPC_combined_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_combined_unified.ipynb)

## New Option
- `recalculate_observer_on_matrix_change`
  - default: `False`
  - valid options: `False | True`

When enabled:
- matrix notebooks recompute `L = compute_observer_gain(A, C, poles)` after each matrix update
- combined notebooks do the same inside the combined loop after the current matrix-adjusted MPC model is set

When disabled:
- the observer gain is computed once, matching the previous behavior

## Notebook Integration
The affected notebooks now:
- expose `RECALCULATE_OBSERVER_ON_MATRIX_CHANGE` as a visible local variable
- print it in the grouped run summary
- pass it into `matrix_cfg` or `combined_cfg`

## Validation
- `py_compile` passed for the touched runner and notebook-parameter modules
- the four affected notebooks parsed successfully with `nbformat` and `ast`
