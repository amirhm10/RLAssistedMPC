# 2026-05-19 archive distillation notebooks replaced by py

## What changed

- moved the five active distillation notebooks that now have `.py` entrypoints into:
  - `archive/distillation_notebooks_replaced_by_py/`

Moved files:

- `distillation_RL_assisted_MPC_markov_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_dueling_unified.ipynb`
- `distillation_RL_assisted_MPC_residual_unified.ipynb`
- `distillation_RL_assisted_MPC_weights_unified.ipynb`

## Notes

- the generated `.py` entrypoints remain at the repo root and are now the intended runnable surfaces for these five distillation workflows
- the original notebooks were archived rather than deleted so they can be restored if needed after script-side validation
