# Unify Distillation RL Reward Profile

## Summary

Unified the active distillation RL reward parameters to one canonical shared profile in `systems/distillation/config.py`.

## Canonical Distillation Reward Values

- `k_rel = [0.3, 0.02]`
- `band_floor_phys = [0.003, 0.3]`
- `tau_frac = 0.7`
- `gamma_in = 0.5`
- `beta = 7.0`
- `lam_in = 1.0`
- `bonus_k = 12.0`
- `reward_scale = 1.0`

## Scope

- Applied only to the active distillation case-study surface.
- Polymer reward defaults were not changed.
- Archived distillation notebooks under `DIstillation Column Case/...` were not modified.

## Notes

The active distillation notebook families already read reward settings from the shared notebook-default path, so no extra family-specific reward overrides were kept.
