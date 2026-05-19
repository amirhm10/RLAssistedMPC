# 2026-05-19 export active distillation notebooks to py

## What changed

- added runnable `.py` siblings for the active distillation notebooks:
  - `distillation_RL_assisted_MPC_markov_unified.py`
  - `distillation_RL_assisted_MPC_horizons_unified.py`
  - `distillation_RL_assisted_MPC_horizons_dueling_unified.py`
  - `distillation_RL_assisted_MPC_residual_unified.py`
  - `distillation_RL_assisted_MPC_weights_unified.py`

## Notes

- the scripts preserve notebook cell order and convert markdown cells into comments
- the Markov script prints `result_bundle["summary_metrics"]` explicitly so the former notebook tail expression still shows output in script mode
- the original notebooks were kept in place; these `.py` files are parallel entrypoints so the user can run them directly without Jupyter

## Validation

- `py_compile` passed for all five generated `.py` scripts in `rl-env`
