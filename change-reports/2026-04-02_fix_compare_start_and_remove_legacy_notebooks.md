# Fix Warm-Start Comparisons And Remove Legacy Notebooks

## Summary

- Fixed the misleading baseline-comparison default in the unified RL notebooks.
- Removed the superseded split and mismatch legacy notebooks now that the unified notebook surface is in place.

## Comparison Fix

- The unified horizon notebook was comparing against MPC starting at episode `1` while `warm_start = 5`.
- That made the comparison overlays look identical because the first compared episodes were still running the baseline/default policy.
- The unified notebooks now compute `COMPARE_START_EPISODE = max(configured_compare_start, warm_start + 1)` before calling the shared comparison path.
- The combined plotting path now accepts `compare_start_episode` separately from `start_episode`, so baseline comparisons can start after warm start without changing the main plotting window.

## Notebook Cleanup

Removed legacy notebooks replaced by unified entrypoints:

- `MPCOffsetFree.ipynb`
- `MPCOffsetFreeDist.ipynb`
- `MPCOffsetFreeDist1.ipynb`
- `MPCOffsetFreeDist2.ipynb`
- `RL_assisted_MPC_horizons.ipynb`
- `RL_assisted_MPC_horizons_dist.ipynb`
- `RL_assisted_MPC_matrices.ipynb`
- `RL_assisted_MPC_matrices_dist.ipynb`
- `RL_assisted_MPC_matrices_SAC.ipynb`
- `RL_assisted_MPC_matrices_dsit_SAC.ipynb`
- `RL_assisted_MPC_matrices_model_mismatch.ipynb`
- `RL_assisted_MPC_weights.ipynb`
- `RL_assisted_MPC_weights_dist.ipynb`
- `RL_assisted_MPC_combined.ipynb`
- `RL_assisted_MPC_combined_dist.ipynb`
- `RL_assisted_MPC_residual_model_mismatch.ipynb`
- `RL_assisted_MPC_residual_model_mismatch1.ipynb`
- `RL_assisted_MPC_residual_model_mismatch2.ipynb`
- `RL_assisted_MPC_residual_model_mismatch_multi.ipynb`

## Docs Updated

- `AGENTS.md`
- `report/notebook_refactor_audit.md`
- `report/model_mismatch_usage.md`

## Validation

- Notebook schema validation for all edited unified notebooks
- Shared plotting import/compile checks
- Confirmed the compare-start override is present in the edited notebooks
