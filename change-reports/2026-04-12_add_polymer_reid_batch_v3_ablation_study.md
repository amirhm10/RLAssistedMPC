# Polymer ReID Batch V3 Ablation Study

## Summary

Added a separate polymer-only v3 ablation surface for online re-identification diagnosis without changing the existing `reid_batch` or `reid_batch_v2` workflows.

## Main Changes

- added `experiments/reid_batch_ablation_matrix.py` to define the Tier 1, Tier 2, and Tier 3 study matrix
- added `experiments/run_reid_batch_ablation_study.py` as the primary reusable execution path
- added `RL_assisted_MPC_reid_batch_v3_ablation_unified.ipynb` as a thin launcher notebook
- extended `utils/reid_batch.py` with:
  - basis metadata for `A` and `B` partitions
  - split regularization and split bound resolution
  - `id_component_mode = "AB" | "A_only" | "B_only"`
  - per-run theta/clipping summary helpers
- extended `utils/reid_batch_runner.py` to save study metadata and resolved split A/B identification settings
- extended `utils/plotting_core.py` and `utils/plotting.py` with:
  - richer per-run theta and model-delta plots
  - study-level ablation summary plotting
- added the polymer defaults family `get_polymer_notebook_defaults("reid_batch_v3_study")`

## Notes

- the observer remains nominal in all v3 runs
- the v3 defaults use the short diagnostic preset and TD3 by default
- legacy `reid_batch` and `reid_batch_v2` notebooks were left unchanged
