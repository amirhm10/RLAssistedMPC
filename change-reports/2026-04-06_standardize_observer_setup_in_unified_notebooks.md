## Summary

Standardized the observer setup pattern across the active baseline and horizon unified notebooks.

## What Changed

- Updated the shared baseline and horizon runners to compute the observer gain `L` internally from `poles` when `L` is not passed explicitly.
- Kept backward compatibility so existing callers that still provide `L` continue to work.
- Updated the visible notebook setup in:
  - `MPCOffsetFree_unified.ipynb`
  - `RL_assisted_MPC_horizons_unified.ipynb`
  - `distillation_MPCOffsetFree_unified.ipynb`
  - `distillation_RL_assisted_MPC_horizons_unified.ipynb`

## Notebook UX

- The notebooks now expose an observer section with the pole vector only.
- The cells include an explicit note that the shared runner computes `L` internally from those poles.
- The notebooks no longer compute `L` inline or pass it through `runtime_ctx`.

## Why

The unified distillation notebooks looked inconsistent because some notebooks visibly computed the observer gain while others only passed poles and relied on shared runners. This change makes the visible structure consistent while preserving the same observer behavior.
