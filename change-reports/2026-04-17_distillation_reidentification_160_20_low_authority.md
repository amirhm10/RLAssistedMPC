# 2026-04-17 distillation reidentification 160_20 low-authority defaults

Adjusted the active distillation reidentification defaults in
`systems/distillation/notebook_params.py` for a follow-up diagnostic run.

This setup keeps the recent open-blend changes in place:

- `candidate_guard_mode = "fro_only"`
- `normalize_blend_extras = False`
- `blend_validity_mode = "off"`
- `blend_validity_scale_floor = 1.0`
- `blend_validity_fallback_scale = 1.0`
- `blend_validity_invalid_candidate_scale = 1.0`
- `freeze_identification_during_warm_start = False`

and changes only the identification cadence plus blend authority:

- `id_window = 160`
- `id_update_period = 20`
- `eta_tau_A = 0.03`
- `eta_tau_B = 0.03`

The goal is to isolate the slower reidentification cadence while keeping
validity suppression disabled, but still giving the blend channel very low
authority through strong eta smoothing.
