# Distillation Rollback, TD3-Only Markov Benchmark, And Residual Guard

Implemented the next distillation experiment setup requested on 2026-06-01.

## Changes

- Restored active single-runner horizon and dueling-horizon grids to the previous narrow range:
  - `Np = 4..14`
  - `Nc = 2..13`
  - 87 valid `Nc <= Np` actions.
- Restored active single-runner weights bounds to `WEIGHT_MULTIPLIER_BOUNDS`, currently `[0.75, 2.0]`.
- Adjusted the weights authority cap ramp from `0.10 -> 1.50` to `0.10 -> 1.00` for the restored bounds.
- Aligned horizon release filtering to the restored grid:
  - protected: `Np 4..12`, `Nc 2..6`
  - ramp: `Np 4..14`, `Nc 2..10`
  - full grid after release: `Np 4..14`, `Nc 2..13`
- Kept reward probation disabled for weights, horizon, dueling horizon, residual, and Markov.
- Added `distillation_RL_assisted_MPC_markov_td3_only_no_safeguard_current_reward_unified.py`.
  - Preserves TD3-only Markov execution behavior with forced TD3, no active LS fallback, no active priority fallback, no active z-safety, and `z_bound = 0.05`.
  - Uses the current high-temperature reward defaults, including `Q_diag = [37000, 20000]`.
  - Uses the historical post-warm BC window toward an LS-action target with `lambda_bc_start = 0.2`, `lambda_bc_end = 0.0`, and 5 active post-warm subepisodes.
  - Saves under separate result and comparison prefixes.
- Added Markov shadow safety logs that do not alter execution:
  - shadow z-safety projection
  - shadow TD3-priority pass/fail phase and authority scale
  - shadow LS priority eligibility
  - shadow nominal fallback eligibility
  - shadow BC handoff authority and correction norm
- Added active residual early-release guard for the first 20 post-warm subepisodes.
  - Computes one-step linear shadow objective and band-normalized predicted error.
  - Shrinks harmful residual moves through `[0.5, 0.25, 0.1, 0.0]`.
  - Applies only step-local shrinkage before physical headroom projection.
  - Keeps rho inactive and reward probation disabled.

## Validation

- `py_compile` passed for:
  - `systems/distillation/notebook_params.py`
  - `utils/residual_runner.py`
  - `utils/markov_runner.py`
  - all five active distillation entrypoints
  - the new TD3-only Markov benchmark entrypoint
- No-Aspen smoke checks passed:
  - reward probation disabled across the five active runners
  - weights bounds restored to `[0.75, 2.0]`
  - standard and dueling horizon defaults build 87 valid recipes and include `(6, 3)`
  - horizon release filter projects during the protected window and accepts the full restored grid later
  - TD3-only Markov benchmark settings force TD3 with active safeguards disabled and shadow safety enabled
  - residual guard shrinks a synthetic harmful residual and leaves a synthetic improving residual unchanged

No full Aspen simulation was run.
