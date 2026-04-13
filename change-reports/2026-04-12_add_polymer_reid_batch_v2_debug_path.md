# Polymer ReID Batch V2 Debug Path

## Summary

Added a separate polymer-only `reid_batch_v2` workflow without changing the
legacy `reid_batch` notebook surface.

## What Changed

- Added richer identification bases in `utils/reid_batch.py`:
  - `scalar_legacy`
  - `rowcol`
  - `block_polymer`
- Extended the shared runner in `utils/reid_batch_runner.py` with:
  - explicit `candidate_guard_mode`
  - `observer_update_alignment`
  - normalized v2 blend-policy extras
  - candidate-vs-active diagnostics
  - theta clipping diagnostics
- Added polymer defaults family `reid_batch_v2` in
  `systems/polymer/notebook_params.py`
- Added a new notebook:
  - `RL_assisted_MPC_reid_batch_v2_unified.ipynb`
- Extended `plot_reid_batch_results(...)` with candidate-vs-active, clipping,
  and reward-vs-eta figures when the traces are present
- Updated the re-identification report in
  `report/05_online_reidentification_batch_ridge_blend.tex`

## Compatibility

- The legacy `RL_assisted_MPC_reid_batch_unified.ipynb` path remains intact.
- Legacy bundles still plot because all new diagnostics are optional.
