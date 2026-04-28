# Distillation Reidentification: Restore RL Blend Authority

## What changed

- Updated `systems/distillation/notebook_params.py` so the distillation
  reidentification defaults no longer force a constant blend.
- Changed:
  - `force_eta_constant: [0.05, 0.05] -> None`

## Why

The April 20, 2026 run `20260420_171837` was a useful isolation for
`id_window = 160`, `id_update_period = 20`, warm-start identification freeze,
and validity-off behavior, but it bypassed actor-controlled blending. Restoring
`force_eta_constant = None` gives the RL policy authority to choose the blend
again while keeping the cleaner warm-start and validity settings.

## Remaining defaults

The following reidentification defaults remain unchanged:

- `id_window = 160`
- `id_update_period = 20`
- `eta_tau_A = eta_tau_B = 1.0`
- `freeze_identification_during_warm_start = True`
- `blend_validity_mode = "off"`
- `candidate_guard_mode = "fro_only"`

So the next run isolates RL-controlled blend behavior under the slower cadence
without the April 17 fade suppression and without the April 18 warm-start drift.
