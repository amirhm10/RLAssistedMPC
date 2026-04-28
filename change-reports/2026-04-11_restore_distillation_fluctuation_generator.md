## Summary

Restored the distillation fluctuation disturbance generator to the older
ramp-based sequence used in prior experiments and made it the default
distillation fluctuation path.

## Changed Files

- `systems/distillation/scenarios.py`
- `distillation_MPCOffsetFree_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_dueling_unified.ipynb`
- `distillation_RL_assisted_MPC_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_structured_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_weights_unified.ipynb`
- `distillation_RL_assisted_MPC_residual_unified.ipynb`
- `distillation_RL_assisted_MPC_combined_unified.ipynb`

## Details

- Replaced the piecewise-constant fluctuation generator with the prior
  slow-target linear-ramp generator plus white-noise jitter.
- Restored the default fluctuation parameters:
  - `slow_horizon_range=(5000, 10000)`
  - `slow_std=2000`
  - `slow_offset_bounds=(-2500, 2500)`
  - `fast_std=50`
  - `seed=42`
- Updated all active distillation notebooks to pass the nominal feed directly
  from the live distillation run via `float(system.feed.FmR.Value)`.

## Validation

- Python AST parse passed for `systems/distillation/scenarios.py`.
- All touched distillation notebook code cells compiled successfully.
- The restored helper matches the prior generator exactly for the same inputs
  and seed (`max_abs_diff = 0.0` for an 80,000-sample test).
- Verification plot saved at:
  - `Distillation/Results/disturbance_debug/distillation_fluctuation_default_80000.png`
