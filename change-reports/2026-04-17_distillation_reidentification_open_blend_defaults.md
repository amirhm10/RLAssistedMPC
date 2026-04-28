# 2026-04-17 distillation reidentification open-blend defaults

Adjusted the active distillation reidentification defaults in
`systems/distillation/notebook_params.py` to remove the blend-validity fade and
give the RL blend much more direct authority for the next run.

- switched `candidate_guard_mode` from `fro_validation_clip` back to `fro_only`
- set `normalize_blend_extras = False`
- disabled blend-validity suppression with `blend_validity_mode = "off"`
- set `blend_validity_scale_floor = 1.0`
- set `blend_validity_fallback_scale = 1.0`
- set `blend_validity_invalid_candidate_scale = 1.0`
- restored the pre-hardening identification cadence:
  - `id_window = 80`
  - `id_update_period = 5`
- restored the more permissive A-side identification limits:
  - `lambda_prev_A = 1e-2`
  - `theta_low_A = -0.15`
  - `theta_high_A = 0.15`
  - `delta_A_max = 0.10`
- removed blend smoothing attenuation by setting:
  - `eta_tau_A = 1.0`
  - `eta_tau_B = 1.0`
- disabled `freeze_identification_during_warm_start`
- neutralized the residual-based blend extras by setting:
  - `blend_extra_clip = 1.0e6`
  - `blend_residual_scale = 1.0e6`

This is intentionally an exploration-oriented configuration. It is meant to let
the distillation reidentification agent use the blend channel directly again,
even if that reintroduces collapse risk.
