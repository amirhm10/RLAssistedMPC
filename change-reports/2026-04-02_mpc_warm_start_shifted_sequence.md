# MPC Warm-Start With Shifted Previous Solution

## What changed

Added a shared MPC warm-start helper and wired it into every active shared runner:

- `utils/helpers.py`
- `utils/mpc_baseline_runner.py`
- `utils/horizon_runner.py`
- `utils/matrix_runner.py`
- `utils/weights_runner.py`
- `utils/residual_runner.py`
- `utils/combined_runner.py`

## Behavior

The MPC optimization no longer starts from an all-zero control sequence on every step.

After each solve, the previous optimal control sequence is shifted forward by one move:

- discard the first control move
- shift the remaining sequence forward
- repeat the final move once

That shifted sequence is used as the initial guess for the next MPC solve.

## Special cases

- Baseline, matrix, weights, and residual use a fixed control horizon, so the warm-start vector persists across all steps.
- Horizon and combined can change `Hc`, so the warm-start vector is reset to zeros whenever the control horizon changes.

## Expected effect

This should reduce MPC solve time noticeably without changing controller semantics.

The most likely gains are in:

- `MPCOffsetFree_unified.ipynb`
- `RL_assisted_MPC_horizons_unified.ipynb`
- `RL_assisted_MPC_matrices_unified.ipynb`
- `RL_assisted_MPC_weights_unified.ipynb`
- `RL_assisted_MPC_residual_unified.ipynb`
- `RL_assisted_MPC_combined_unified.ipynb`

and the matching distillation unified notebooks, since they use the same shared runners.
