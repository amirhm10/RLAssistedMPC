# 2026-04-18 distillation reidentification 160_20 fixed small blend defaults

Adjusted the active distillation reidentification defaults in
`systems/distillation/notebook_params.py` for a cleaner isolation study of the
`id_window = 160`, `id_update_period = 20` setting.

This setup keeps:

- `candidate_guard_mode = "fro_only"`
- `normalize_blend_extras = False`
- `blend_validity_mode = "off"`
- `blend_validity_scale_floor = 1.0`
- `blend_validity_fallback_scale = 1.0`
- `blend_validity_invalid_candidate_scale = 1.0`

and changes the blend path to avoid both hidden warm-start drift and the
ambiguous "low authority via eta smoothing" behavior:

- `freeze_identification_during_warm_start = True`
- `force_eta_constant = [0.05, 0.05]`
- `eta_tau_A = 1.0`
- `eta_tau_B = 1.0`

This means the next run will use a small fixed blend after warm start while the
reidentification cadence stays at `160/20`.

Important limitation:

- this isolates the identification cadence under a small blend
- it does **not** test RL blend-action selection, because `force_eta_constant`
  bypasses the actor's dual-eta output

If a true RL-controlled low-authority study is needed, the code surface needs a
separate configurable max-eta cap rather than only `eta_tau` or
`force_eta_constant`.
