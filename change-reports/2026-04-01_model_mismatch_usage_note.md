# Model Mismatch Usage Note

## Summary

- Added a markdown explainer for the legacy mismatch notebooks.
- The note documents how mismatch is injected into the plant and how it is exposed to the RL policies.

## Added File

- `report/model_mismatch_usage.md`

## Main Points Covered

- difference between the matrix mismatch notebook and the residual mismatch family
- how `Qi`, `Qs`, `hA`, and `CMf` are used to create plant-side mismatch
- how innovation and tracking-error terms are appended to the RL state in the legacy mismatch notebooks
- how residual mismatch notebooks use mismatch-aware clipping and replay the executed residual move
- how the `multi` mismatch notebook combines residual correction with matrix scaling
- how the new unified notebooks differ from the legacy mismatch approach
