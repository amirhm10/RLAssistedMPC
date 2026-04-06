## Summary

This change aligns the canonical distillation reward defaults with the archived distillation horizon notebook.

## What changed

- Updated `systems/distillation/config.py` so `RL_REWARD_DEFAULTS` matches the archived `RL_assisted_MPC_horizons.ipynb` reward defaults more closely.
- The concrete mismatch fixed here was:
  - `beta: 7.0 -> 5.0`

## Consistency check

All active distillation runtime notebooks already build their shared reward from `RL_REWARD_DEFAULTS`, so this single config change propagates to:

- `distillation_MPCOffsetFree_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_unified.ipynb`
- `distillation_RL_assisted_MPC_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_weights_unified.ipynb`
- `distillation_RL_assisted_MPC_residual_unified.ipynb`
- `distillation_RL_assisted_MPC_combined_unified.ipynb`

## Notes

- `distillation_systemIdentification_unified.ipynb` does not build an RL reward and is unaffected.
- This change follows the user decision to use the archived distillation horizon reward setup as the default distillation reward profile.
