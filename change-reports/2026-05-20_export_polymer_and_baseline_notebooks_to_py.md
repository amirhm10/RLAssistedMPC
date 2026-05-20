# Export Polymer And Baseline Notebooks To Py

Date: 2026-05-20

## Summary

- Exported the requested notebooks to runnable root-level `.py` scripts.
- Moved the original notebooks into `archive/notebooks_replaced_by_py_20260520/`.
- Preserved notebook cell order, converted markdown cells to comments, and kept code cells executable.

## Exported Scripts

- `distillation_MPCOffsetFree_unified.py`
- `MPCOffsetFree_unified.py`
- `RL_assisted_MPC_weights_unified.py`
- `RL_assisted_MPC_markov_unified.py`
- `RL_assisted_MPC_residual_unified.py`
- `RL_assisted_MPC_horizons_dueling_unified.py`
- `RL_assisted_MPC_horizons_unified.py`

## Archived Notebooks

- `distillation_MPCOffsetFree_unified.ipynb`
- `MPCOffsetFree_unified.ipynb`
- `RL_assisted_MPC_weights_unified.ipynb`
- `RL_assisted_MPC_markov_unified.ipynb`
- `RL_assisted_MPC_residual_unified.ipynb`
- `RL_assisted_MPC_horizons_dueling_unified.ipynb`
- `RL_assisted_MPC_horizons_unified.ipynb`

## Validation

- Ran `py_compile` on all seven generated `.py` files in `rl-env`.
