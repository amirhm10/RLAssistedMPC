# Add MPC Warm-Start Toggle To All Unified Notebooks

## Summary
- Added `USE_SHIFTED_MPC_WARM_START` to the remaining unified notebooks.
- Set the default to `False` everywhere so unified runs match the legacy zero-initialized MPC solve unless explicitly overridden.
- Extended the shared baseline, matrix, weights, residual, and combined runners to honor the toggle.

## Files
- `utils/horizon_runner.py`
- `utils/mpc_baseline_runner.py`
- `utils/matrix_runner.py`
- `utils/weights_runner.py`
- `utils/residual_runner.py`
- `utils/combined_runner.py`
- `MPCOffsetFree_unified.ipynb`
- `RL_assisted_MPC_horizons_unified.ipynb`
- `RL_assisted_MPC_matrices_unified.ipynb`
- `RL_assisted_MPC_weights_unified.ipynb`
- `RL_assisted_MPC_residual_unified.ipynb`
- `RL_assisted_MPC_combined_unified.ipynb`
- `distillation_MPCOffsetFree_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_unified.ipynb`
- `distillation_RL_assisted_MPC_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_weights_unified.ipynb`
- `distillation_RL_assisted_MPC_residual_unified.ipynb`
- `distillation_RL_assisted_MPC_combined_unified.ipynb`

## Validation
- Parsed all active unified notebooks with `ast.parse`.
- Compiled the touched shared runners in `rl-env`.
