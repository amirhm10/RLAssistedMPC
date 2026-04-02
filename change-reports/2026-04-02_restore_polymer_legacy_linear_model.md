# Restore Polymer Legacy Linear Model

## Summary
- Restored the polymer unified notebooks to load the legacy polymer linear model from `Data/system_dict`.
- Left distillation on its canonical `system_dict.pickle` path.

## Root Cause
- The legacy polymer notebooks loaded `Data/system_dict` without an extension.
- The unified polymer notebooks were loading `Data/system_dict.pickle`.
- Those two files are not equivalent in this repo.

## Impact
- `Data/system_dict` reproduces the legacy first nominal MPC move:
  - `u = [471.97248561, 387.109175]`
- `Data/system_dict.pickle` produces the newer aggressive first move:
  - `u = [542.48489569, 547.96192256]`
- This difference changes the polymer closed-loop trajectory and therefore changes both:
  - shared reward traces
  - quadratic MPC reward traces

## Files
- `MPCOffsetFree_unified.ipynb`
- `RL_assisted_MPC_horizons_unified.ipynb`
- `RL_assisted_MPC_matrices_unified.ipynb`
- `RL_assisted_MPC_weights_unified.ipynb`
- `RL_assisted_MPC_residual_unified.ipynb`
- `RL_assisted_MPC_combined_unified.ipynb`
