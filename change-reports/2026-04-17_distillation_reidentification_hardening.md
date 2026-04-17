# Distillation Reidentification Hardening

- hardened the distillation reidentification defaults in `systems/distillation/notebook_params.py`:
  - increased `id_window` from `80` to `160`
  - increased `id_update_period` from `5` to `20`
  - tightened the `A` side with `lambda_prev_A: 1e-2 -> 5e-2`, `theta_low_A/theta_high_A: [-0.15, 0.15] -> [-0.10, 0.10]`, and `delta_A_max: 0.10 -> 0.07`
  - reduced blend exposure speed with `eta_tau_A/B: 0.1 -> 0.03`
  - kept `observer_refresh_enabled=False`
  - enabled a stricter distillation guard mode plus validation and blend-fade thresholds
- updated the shared low-rank reidentification helper in `utils/reidentification.py`:
  - added optional train/validation residual diagnostics to the batch solve path
  - extended candidate selection so non-`fro_only` guard modes can reject candidates on clipping, conditioning, and validation/full residual checks
  - added a validity-aware blend scaling helper that can fade the learned model contribution when identification quality degrades
- updated `utils/reidentification_runner.py`:
  - added `freeze_identification_during_warm_start`
  - applied the new guard configuration when selecting candidates
  - applied validity-aware scaling to the dual-eta blend before building the prediction model
  - saved extra diagnostics for requested blend values, validity scales, and residual ratios
- updated `distillation_RL_assisted_MPC_reidentification_unified.ipynb` so the new distillation guard and blend-fade fields are resolved, summarized, and passed into `reid_cfg`

Validation:

- `C:\\Users\\HAMEDI\\miniconda3\\envs\\rl-env\\python.exe -m py_compile utils\\reidentification.py utils\\reidentification_runner.py systems\\distillation\\notebook_params.py`
- notebook JSON parse and code-cell compile for `distillation_RL_assisted_MPC_reidentification_unified.ipynb`
- lightweight helper smoke test for `compute_blend_validity_scale(...)` using the distillation defaults
